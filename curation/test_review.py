"""HTTP-contract and provenance tests; no external service, credentials or paid calls."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import unittest

from curation.review import Config, PROMPT, input_identity, review_one, sha, validate_review, request_id, parse_response
from curation.schema import VERSION


def row():
    return dict(schema_version=VERSION, id='recording:segment-1', source_id='fixture',
        recording_id='recording', audio_path='/not-sent/private.flac', audio_sha256='a'*64,
        sentence='Actually, na-miss ko ang pagkaon sa balay.',
        prompt='SECRET SOURCE QUESTION: diin ka gusto magkadto?', text_is_prompt=False,
        language='hil', languages=['hil','eng'], speech_type='spontaneous', speaker_id='PRIVATE_SPEAKER',
        duration=5.0, num_words=8, split='unassigned', review_status='draft', asr_status='transcribed',
        rights=dict(training='unknown',commercial='unknown',redistribution='unknown'),
        provenance={'asr_config_sha256':'c'*64})


def valid_review(r):
    return dict(id=r['id'], triage='no_obvious_text_issue',
                language_assessment='Understandable Hiligaynon-English code-switching; no audio available.', issues=[])


def envelope(value):
    return dict(model='test-returned-model', choices=[dict(finish_reason='stop',message={'content':json.dumps(value)})],
                usage={'prompt_tokens':10,'completion_tokens':10,'total_tokens':20})


@contextmanager
def server(responses):
    requests=[]
    queue=list(responses)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'body':body,'path':self.path,'authorization':self.headers.get('Authorization')})
            status,data=queue.pop(0) if queue else (500,{'error':'unexpected extra call'})
            data=data if isinstance(data,bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self,*args): pass
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=http.serve_forever,daemon=True)
    thread.start()
    try:
        yield Config(base_url=f'http://127.0.0.1:{http.server_port}/v1',retry_delay=0,timeout=2),requests
    finally:
        http.shutdown();http.server_close();thread.join()


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=Path(self.tmp.name)
        self.r=row()
        self.r['num_words']=len(self.r['sentence'].split())
    def tearDown(self): self.tmp.cleanup()

    def test_long_id_uses_bound_short_wire_identity_and_legacy_still_parses(self):
        self.r['id']='a'*64
        short=request_id(self.r,'short-hash-if-long-v1')
        self.assertEqual(len(short),13)
        parsed=valid_review(self.r);parsed['id']=short
        with server([(200,envelope(parsed))]) as (cfg,requests):
            result=review_one(self.r,cfg,'test-key',self.out)
            sent=json.loads(requests[0]['body']['messages'][1]['content'])
            self.assertEqual(sent['id'],short)
            self.assertEqual(result['review']['id'],self.r['id'])
            self.assertEqual(result['input_sha256'],sha(input_identity(self.r)))
        self.assertEqual(parse_response(envelope(valid_review(self.r)),self.r)['id'],self.r['id'])
        parsed['id']='r000000000000'
        with self.assertRaisesRegex(ValueError,'identity'):
            parse_response(envelope(parsed),self.r,'short-hash-if-long-v1')

    def test_lexical_replacement_is_not_an_approved_suggestion(self):
        review=valid_review(self.r)
        review['triage']='needs_audio_review'
        review['issues']=[dict(quote='pagkaon',kind='unclear_text',suggestion='pagkadto',reason='A guess')]
        with self.assertRaisesRegex(ValueError,'lexical content'):
            validate_review(review,self.r)
        review['issues'][0].update(quote='na-miss',suggestion='na miss')
        validate_review(review,self.r)

    def test_question_exclusion_codemix_prompt_and_cache(self):
        with server([(200,envelope(valid_review(self.r)))]) as (cfg,requests):
            before=json.dumps(self.r,sort_keys=True)
            result=review_one(self.r,cfg,'fake-test-key',self.out)
            second=review_one(self.r,cfg,'fake-test-key',self.out)
            self.assertEqual(result,second)
            self.assertEqual(len(requests),1)
            self.assertEqual(requests[0]['path'],'/v1/chat/completions')
            self.assertEqual(requests[0]['authorization'],'Bearer fake-test-key')
            body=requests[0]['body']
            user=json.loads(body['messages'][1]['content'])
            self.assertEqual(user,{'id':self.r['id'],'transcript':self.r['sentence'],'source_language_hint':self.r['language']})
            wire=json.dumps(body)
            for value in [self.r['prompt'],self.r['speaker_id'],self.r['audio_path']]:
                self.assertNotIn(value,wire)
            self.assertIn('code-switching are valid grammatical constructs',body['messages'][0]['content'])
            self.assertIn('Assume understandable English is an intentional code switch',PROMPT)
            self.assertFalse(result['audio_verified'])
            self.assertFalse(result['human_approved'])
            self.assertEqual(result['status'],'text_review_only')
            self.assertEqual(json.dumps(self.r,sort_keys=True),before)
            for p in self.out.glob('*'):
                self.assertNotIn('fake-test-key',p.read_text())

    def test_cache_invalidation_full_text_audio_model_base_prompt_version(self):
        variants=[]
        for change in [{'sentence':'I really miss ang pagkaon sa balay.'}, {'audio_sha256':'b'*64}]:
            r={**self.r,**change};r['num_words']=len(r['sentence'].split());variants.append(r)
        with server([(200,envelope(valid_review(self.r)))]*6) as (cfg,requests):
            results=[review_one(self.r,cfg,'key',self.out)]
            for r in variants: results.append(review_one(r,cfg,'key',self.out))
            results.append(review_one(self.r,replace(cfg,model='different-model'),'key',self.out))
            results.append(review_one(self.r,replace(cfg,prompt_version='v2'),'key',self.out))
            results.append(review_one(self.r,replace(cfg,base_url=cfg.base_url+'/other'),'key',self.out))
            self.assertEqual(len(requests),6)
            self.assertEqual(len({r['cache_key'] for r in results}),6)

    def test_exact_quote_and_separate_suggestion(self):
        parsed=valid_review(self.r)
        parsed.update(triage='needs_audio_review',issues=[dict(quote='na-miss',kind='unclear_text',
            suggestion='na miss',reason='Tentative spacing only; check audio.')])
        with server([(200,envelope(parsed))]) as (cfg,requests):
            result=review_one(self.r,cfg,'key',self.out)
            self.assertEqual(result['review']['issues'][0]['suggestion'],'na miss')
            self.assertIn('na-miss',self.r['sentence'])

    def test_invented_quote_fails_closed_and_saved_response_is_not_rebilled(self):
        parsed=valid_review(self.r)
        parsed.update(triage='needs_audio_review',issues=[dict(quote='invented words',kind='unclear_text',
            suggestion=None,reason='Suspicious')])
        with server([(200,envelope(parsed))]) as (cfg,requests):
            for _ in range(2):
                with self.assertRaises(ValueError): review_one(self.r,cfg,'key',self.out)
            self.assertEqual(len(requests),1)
            self.assertEqual(len(list(self.out.glob('*.response.json'))),1)
            self.assertEqual(len(list(self.out.glob('*.error.json'))),1)
            self.assertEqual(len(list(self.out.glob('*.json'))),2)
            self.assertTrue((self.out/'errors.jsonl').exists())

    def test_transient_http_retry_and_ledger(self):
        with server([(503,{'error':'busy'}),(429,{'error':'rate limit'}),(200,envelope(valid_review(self.r)))]) as (cfg,requests):
            result=review_one(self.r,cfg,'key',self.out)
            self.assertEqual(len(requests),3)
            errors=[json.loads(x) for x in (self.out/'errors.jsonl').read_text().splitlines()]
            self.assertEqual([e['http_status'] for e in errors],[503,429])
            self.assertEqual(result['review']['triage'],'no_obvious_text_issue')

    def test_payment_error_not_retried(self):
        with server([(402,{'error':'provider credits'})]) as (cfg,requests):
            with self.assertRaisesRegex(RuntimeError,'HTTP 402'): review_one(self.r,cfg,'key',self.out)
            self.assertEqual(len(requests),1)

    def test_retry_bound(self):
        with server([(503,{})]*5) as (cfg,requests):
            with self.assertRaises(RuntimeError): review_one(self.r,cfg,'key',self.out)
            self.assertEqual(len(requests),3)

    def test_invalid_json_and_truncated_completion_fail_closed(self):
        cases=[b'not json',envelope(valid_review(self.r))]
        cases[1]['choices'][0]['finish_reason']='length'
        for i,response in enumerate(cases):
            out=self.out/str(i)
            with server([(200,response)]) as (cfg,requests):
                with self.assertRaises(ValueError): review_one(self.r,cfg,'key',out)
                with self.assertRaises(ValueError): review_one(self.r,cfg,'key',out)
                self.assertEqual(len(requests),1)
                self.assertTrue(list(out.glob('*.response.json')))

    def test_pending_and_empty_transcripts_do_not_request(self):
        with server([]) as (cfg,requests):
            for change in [{'asr_status':'pending'}, {'sentence':'','num_words':0}]:
                with self.assertRaises(ValueError): review_one({**self.r,**change},cfg,'key',self.out)
            self.assertEqual(requests,[])

    def test_cache_requires_raw_response_integrity(self):
        with server([(200,envelope(valid_review(self.r)))]) as (cfg,requests):
            result=review_one(self.r,cfg,'key',self.out)
            (self.out/(result['cache_key']+'.response.json')).write_text('{}')
            with self.assertRaises(ValueError): review_one(self.r,cfg,'key',self.out)
            self.assertEqual(len(requests),1)

    def test_parallel_local_workers_share_one_paid_result(self):
        with server([(200,envelope(valid_review(self.r)))]) as (cfg,requests):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda _:review_one(self.r,cfg,'key',self.out),range(2)))
            self.assertEqual(results[0],results[1])
            self.assertEqual(len(requests),1)

    def test_cli_stops_after_consecutive_failures(self):
        manifest=self.out/'inputs.jsonl'
        rows=[dict(self.r,id=f'item-{i}') for i in range(3)]
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        key=self.out/'key';key.write_text('fake-test-key')
        with server([(503,{}),(503,{}),(503,{})]) as (cfg,requests):
            result=subprocess.run([sys.executable,'-m','curation.review','--manifest',str(manifest),
                '--output',str(self.out/'results'),'--key-file',str(key),'--base-url',cfg.base_url,
                '--max-attempts','1','--max-consecutive-failures','2'],capture_output=True,timeout=10)
            self.assertEqual(result.returncode,1)
            self.assertEqual(len(requests),2)
        summary=json.loads((self.out/'results/summary.json').read_text())
        self.assertTrue(summary['stopped_early'])
        self.assertEqual(summary['selected'],2)
        self.assertEqual(len(summary['failed']),2)

    def test_insecure_remote_url_and_embedded_credential_rejected(self):
        for url in ['http://example.org/v1','https://key@example.org/v1','https://example.org/v1?key=secret']:
            with self.assertRaises(ValueError): replace(Config(),base_url=url).check()


if __name__=='__main__': unittest.main()
