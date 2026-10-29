// Optional reconstruction of the native PowerPoint figure. Empirical reproduction
// does not need this authoring dependency; edit the supplied PPTX directly instead.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));
const target=process.argv[2];
if(!target)throw new Error('Usage: node figure_assets/build_pptx.mjs /new/output.pptx');
try{await fs.access(target);throw new Error('Choose a new path; output already exists.');}catch(e){if(e.code!=='ENOENT')throw e;}
const spec=JSON.parse(await fs.readFile(path.join(here,'Figure_1_shapes.json'),'utf8'));
const p=Presentation.create({slideSize:{width:spec.width,height:spec.height}});
const slide=p.slides.add();slide.background.fill='#FFFFFF';
for(const r of spec.shapes){
 if(r.kind==='shape')slide.shapes.add({geometry:r.geo,name:r.name,position:{left:r.x,top:r.y,width:r.w,height:r.h},fill:r.fill,...(r.shadow?{shadow:r.shadow}:{}),line:{fill:r.stroke,width:r.lw,style:'solid'}});
 else if(r.kind==='text'){
  const s=slide.shapes.add({geometry:'textbox',name:r.name,position:{left:r.x,top:r.y,width:r.w,height:r.h},fill:'none',line:{fill:'none',width:0}});
  s.text=r.t;s.text.style={typeface:spec.font,fontSize:r.size,color:r.color,bold:r.bold,alignment:r.align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:0};
 }else if(r.kind==='line')slide.shapes.add({geometry:'connector',kind:'straight',name:r.name,position:{left:Math.min(r.x1,r.x2),top:Math.min(r.y1,r.y2),width:Math.abs(r.x2-r.x1),height:Math.abs(r.y2-r.y1),horizontalFlip:r.x2<r.x1,verticalFlip:r.y2<r.y1},line:{fill:r.color,width:r.lw,style:r.dash?'dashed':'solid'},tail:{type:r.arrow?'triangle':'none',width:'sm',length:'sm'}});
 else if(r.kind==='poly'){
  const xs=r.points.map(v=>v[0]),ys=r.points.map(v=>v[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;
  slide.shapes.add({geometry:'custom',name:r.name,position:{left:x,top:y,width:w,height:h},fill:r.fill,...(r.shadow?{shadow:r.shadow}:{}),line:{fill:r.stroke,width:r.lw,style:'solid'},customPaths:[{width:w,height:h,commands:[{moveTo:{x:r.points[0][0]-x,y:r.points[0][1]-y}},...r.points.slice(1).map(v=>({lineTo:{x:v[0]-x,y:v[1]-y}})),{close:{}}]}]});
 }else throw new Error('Unknown native shape kind: '+r.kind);
}
slide.speakerNotes.text=spec.notes;
await fs.mkdir(path.dirname(path.resolve(target)),{recursive:true});
await (await PresentationFile.exportPptx(p)).save(target);
console.log('Wrote native editable figure: '+target);
