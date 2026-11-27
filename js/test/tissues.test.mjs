// End-to-end public tissue API: actual GPU, workers, encoding and geometry.
import assert from 'node:assert/strict'
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs'
import {dirname,join,resolve,extname} from 'node:path'
import {fileURLToPath} from 'node:url'
import {createRequire} from 'node:module'
import {createServer} from 'node:http'
import {spawnSync} from 'node:child_process'
const here=dirname(fileURLToPath(import.meta.url)),root=resolve(here,'../..'),dist=join(root,'js/dist')
const [refsArg,outArg]=process.argv.slice(2)
if(!refsArg||!outArg)throw Error('usage: tissues.test.mjs CPU_REFERENCE_DIRECTORY OUTPUT_DIRECTORY')
const refs=resolve(refsArg),out=resolve(outArg);mkdirSync(out,{recursive:true})
const selected=process.env.BC_TEST_CASES?.split(',')
const cases=JSON.parse(readFileSync(join(refs,'references.json'))).filter(c=>!selected||selected.includes(c.key))
assert(cases.length>0,'no matching tissue test cases')
const require=createRequire(join(process.env.BC_PLAYWRIGHT_ROOT||resolve(root,'../brainchop-test'),'package.json'))
const playwright=require('playwright'),received=new Map()
const browserName=process.env.BC_TEST_BROWSER||'chromium'
const files=new Map(['/t1_crop.nii.gz','/t1_2mm.nii.gz'].map(url=>[url,join(root,'test',url.slice(1))]))
const requestedBackends=(process.env.BC_TEST_BACKENDS||'webgpu,webgl2').split(',')
for(const name of ['index.js','worker.js','models/mindmap.bin','models/mindmap-fast.bin',
  ...(requestedBackends.includes('webgl2')?['mindmap','mindmap-fast'].flatMap(m=>['.js','.wasm'].map(e=>`brainchop-${m}-gl${e}`)):[]),
  ...(requestedBackends.includes('cpu')?['mindmap','mindmap-fast'].flatMap(m=>['.js','.wasm'].map(e=>`brainchop-${m}${e}`)):[])])files.set('/'+name,join(dist,name))
const server=createServer((req,res)=>{
  if(req.method==='POST') {const chunks=[];req.on('data',x=>chunks.push(x));req.on('end',()=>{received.set(req.url,Buffer.concat(chunks));res.writeHead(204);res.end()});return}
  res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp')
  if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>Tissue API test</title>');return}
  const p=files.get(req.url);if(!p||!existsSync(p)){res.writeHead(404);res.end();return}
  res.setHeader('Content-Type',extname(p)==='.js'?'text/javascript':extname(p)==='.wasm'?'application/wasm':'application/octet-stream');res.end(readFileSync(p))
})
await new Promise(r=>server.listen(0,'127.0.0.1',r))
const browser=await playwright[browserName].launch({headless:process.env.BC_WGSL_HEADLESS==='1',
  ...(browserName==='chromium'?{...(process.env.BC_CHROME_EXECUTABLE ? {executablePath:process.env.BC_CHROME_EXECUTABLE} : {channel:process.env.BC_CHROME_CHANNEL||'chrome'}),args:['--enable-unsafe-webgpu',...JSON.parse(process.env.BC_CHROME_ARGS||'[]')]}: {})})
const report=[]
try {
  const page=await browser.newPage();page.on('console',m=>console.log('browser:',m.text()));await page.goto(`http://127.0.0.1:${server.address().port}`)
  for(const backend of (process.env.BC_TEST_BACKENDS||'webgpu,webgl2').split(',')) {
  const hardware=await page.evaluate(async backend=>{
    if(backend==='cpu')return {logicalCores:navigator.hardwareConcurrency}
    const {checkSupport,checkWebgl2Support}=await import('/index.js')
    if(backend==='webgl2') {
      const {supported,software,renderer}=checkWebgl2Support('mindmap')
      if(!supported||software)throw Error(`hardware WebGL2 required (${renderer})`)
      return {renderer}
    }
    if((await checkSupport('mindmap')).software)throw Error('software GPU')
    const a=await navigator.gpu?.requestAdapter();if(!a)throw Error('hardware WebGPU required')
    const i=a.info
    return {vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description}
  },backend)
  for(const [index,c] of cases.entries()) {
    const run=await page.evaluate(async({c,index,backend})=>{
      const {segmentTissues}=await import('/index.js'),input=await(await fetch('/'+c.fixture+'.nii.gz')).arrayBuffer()
      const before=input.byteLength,log=[],start=performance.now()
      const result=await segmentTissues(input,{model:c.model,backend,saveConform:c.conform,
        worker:index%2===1,gzipOutput:index%2===0,onLog:l=>{log.push(l);console.log(l)}})
      if(input.byteLength!==before)throw Error('input buffer was detached')
      for(const [t,bytes] of Object.entries(result.tissues))await fetch('/'+t,{method:'POST',body:bytes})
      return {backend:result.backend,worker:result.ranInWorker,elapsedMs:result.elapsedMs,totalMs:performance.now()-start,log}
    },{c,index,backend})
    assert.equal(run.backend,backend);assert.equal(run.worker,index%2===1)
    const paths={}
    for(const t of ['gm','wm','csf','brain']){
      const bytes=received.get('/'+t);assert(bytes?.length>352)
      assert.equal(bytes[0]===31&&bytes[1]===139,index%2===0)
      paths[t]=join(out,backend+'-'+c.key+'_'+t+(index%2===0?'.nii.gz':'.nii'));writeFileSync(paths[t],bytes)
    }
    const compare=spawnSync('python3',['-c',
      'import sys,json;sys.path.insert(0,sys.argv[1]);from cuda_pve_check import compare;refs=json.loads(sys.argv[2]);got=json.loads(sys.argv[3]);print(json.dumps({t:compare(refs[t],got[t]) for t in refs}))',
      join(root,'test'),JSON.stringify(Object.fromEntries(Object.entries(c.files).map(([t,f])=>[t,join(refs,f)]))),JSON.stringify(paths)],{encoding:'utf8'})
    assert.equal(compare.status,0,compare.stderr)
    report.push({case:c.key,...run,hardware,browser:browser.version(),numerical:JSON.parse(compare.stdout)})
    writeFileSync(join(out,'results.json'),JSON.stringify(report,null,2)+'\n');console.log(c.key,compare.stdout.trim())
  }
  }
  const refusals=await page.evaluate(async()=>{
    const {segmentTissues}=await import('/index.js'),results=[]
    for(const opts of [{model:'mindgrab'},{model:'mindmap-fast'},{model:'16chan18cls'},{model:'mindmap',legacyCleanup:true}]) {
      try{await segmentTissues(new Uint8Array([1,2,3,4]),opts);results.push('unexpected success')}
      catch(e){results.push(e.code)}
    }
    return results
  })
  assert.deepEqual(refusals,['unsupported-option','unsupported-option','unsupported-option','unsupported-option'])
  console.log(`Tissue API: ${report.length} selected cases, encoding/worker/geometry checks and option refusals PASS`)
}finally{await browser.close();server.close()}
