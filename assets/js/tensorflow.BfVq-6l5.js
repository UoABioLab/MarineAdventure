import{g as Ic}from"./mediapipe-pose.CqkleIqs.js";function Rc(n,e){for(var t=0;t<e.length;t++){const s=e[t];if(typeof s!="string"&&!Array.isArray(s)){for(const r in s)if(r!=="default"&&!(r in n)){const o=Object.getOwnPropertyDescriptor(s,r);o&&Object.defineProperty(n,r,o.get?o:{enumerable:!0,get:()=>s[r]})}}}return Object.freeze(Object.defineProperty(n,Symbol.toStringTag,{value:"Module"}))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tc=1e-7,Ec=1e-4;class Nc{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class Eo{refCount(e){return pe("refCount")}incRef(e){return pe("incRef")}timerAvailable(){return!0}time(e){return pe("time")}read(e){return pe("read")}readSync(e){return pe("readSync")}readToGPU(e,t){return pe("readToGPU")}numDataIds(){return pe("numDataIds")}disposeData(e,t){return pe("disposeData")}write(e,t,s){return pe("write")}move(e,t,s,r,o){return pe("move")}createTensorFromGPUData(e,t,s){return pe("createTensorFromGPUData")}memory(){return pe("memory")}floatPrecision(){return pe("floatPrecision")}epsilon(){return this.floatPrecision()===32?Tc:Ec}dispose(){return pe("dispose")}}function pe(n){throw new Error(`'${n}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bn(n,e,t){return Math.max(n,Math.min(e,t))}function Hs(n){return n%2===0?n:n+1}function rn(n,e,t){const s=n[e];n[e]=n[t],n[t]=s}function kc(n){let e=0;for(let t=0;t<n.length;t++)e+=n[t];return e}function I(n,e){if(!n)throw new Error(typeof e=="string"?e:e())}function No(n,e,t=""){I(J(n,e),()=>t+` Shapes ${n} and ${e} must match`)}function E(n){if(n.length===0)return 1;let e=n[0];for(let t=1;t<n.length;t++)e*=n[t];return e}function J(n,e){if(n===e)return!0;if(n==null||e==null||n.length!==e.length)return!1;for(let t=0;t<n.length;t++)if(n[t]!==e[t])return!1;return!0}function Mn(n){return n%1===0}function $s(n){const e=Math.ceil(Math.sqrt(n));return[e,Math.ceil(n/e)]}function _t(n,e){return e<=n.length?n:n+" ".repeat(e-n.length)}function vr(n,e=r=>0,t,s){return new Promise((r,o)=>{let i=0;const a=()=>{if(n()){r();return}i++;const c=e(i);if(t!=null&&i>=t){o();return}s!=null?s(a,c):setTimeout(a,c)};a()})}function Ac(n,e){let t=1,s=-1;for(let o=0;o<n.length;++o)if(n[o]>=0)t*=n[o];else if(n[o]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${o}`);s=o}else if(n[o]<0)throw Error(`Shapes can not be < 0. Found ${n[o]} at dim ${o}`);if(s===-1){if(e>0&&e!==t)throw Error(`Size(${e}) must match the product of shape ${n}`);return n}if(t===0)throw Error(`Cannot infer the missing size in [${n}] when there are 0 elements`);if(e%t!==0)throw Error(`The implicit shape can't be a fractional number. Got ${e} / ${t}`);const r=n.slice();return r[s]=e/t,r}function de(n,e){const t=e.length;return n=n==null?e.map((s,r)=>r):[].concat(n),I(n.every(s=>s>=-t&&s<t),()=>`All values in axis param must be in range [-${t}, ${t}) but got axis ${n}`),I(n.every(s=>Mn(s)),()=>`All values in axis param must be integers but got axis ${n}`),n.map(s=>s<0?t+s:s)}function yt(n,e){const t=[],s=[];for(let r=0;r<n.length;++r)n[r]!==1&&(t.push(n[r]),s.push(r));return{newShape:t,keptDims:s}}function pt(n,e){return q(n,e)}function q(n,e){let t=null;if(n==null||n==="float32")t=new Float32Array(e);else if(n==="int32")t=new Int32Array(e);else if(n==="bool")t=new Uint8Array(e);else if(n==="string")t=new Array(e);else throw new Error(`Unknown data type ${n}`);return t}function Fc(n,e){for(let t=0;t<n.length;t++){const s=n[t];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${e} being uploaded contains ${s}.`)}}function Dc(n){return n==="bool"||n==="complex64"||n==="float32"||n==="int32"||n==="string"}function Oc(n,e){return!(e==="complex64"||e==="float32"&&n!=="complex64"||e==="int32"&&n!=="float32"&&n!=="complex64"||e==="bool"&&n==="bool")}function Vn(n){if(n==="float32"||n==="int32")return 4;if(n==="complex64")return 8;if(n==="bool")return 1;throw new Error(`Unknown dtype ${n}`)}function Pc(n){if(n==null)return 0;let e=0;return n.forEach(t=>e+=t.length),e}function Zn(n){return typeof n=="string"||n instanceof String}function _c(n){return typeof n=="boolean"}function Lc(n){return typeof n=="number"}function xn(n){return Array.isArray(n)?xn(n[0]):n instanceof Float32Array?"float32":n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray?"int32":Lc(n)?"float32":Zn(n)?"string":_c(n)?"bool":"float32"}function vs(n){return!!(n&&n.constructor&&n.call&&n.apply)}function Ss(n,e){for(let t=e;t<n;++t)if(n%t===0)return t;return n}function Z(n){const e=n.length;if(e<2)return[];const t=new Array(e-1);t[e-2]=n[e-1];for(let s=e-3;s>=0;--s)t[s]=t[s+1]*n[s+1];return t}function ko(n,e,t,s=!1){const r=new Array;if(e.length===1){const o=e[0]*(s?2:1);for(let i=0;i<o;i++)r[i]=t[n+i]}else{const o=e[0],i=e.slice(1),a=i.reduce((c,l)=>c*l)*(s?2:1);for(let c=0;c<o;c++)r[c]=ko(n+c*a,i,t,s)}return r}function Sr(n,e,t=!1){if(n.length===0)return e[0];const s=n.reduce((r,o)=>r*o)*(t?2:1);if(s===0)return[];if(s!==e.length)throw new Error(`[${n}] does not match the input size ${e.length}${t?" for a complex tensor":""}.`);return ko(0,n,e,t)}function Bc(n,e){const t=nt(n,e);for(let s=0;s<t.length;s++)t[s]=1;return t}function nt(n,e){if(e==null||e==="float32"||e==="complex64")return new Float32Array(n);if(e==="int32")return new Int32Array(n);if(e==="bool")return new Uint8Array(n);throw new Error(`Unknown data type ${e}`)}function Cn(n){n.forEach(e=>{I(Number.isInteger(e)&&e>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${n}].`)})}function Is(n,e,t){if(e===0)return 0;if(e===1)return n[0];let s=n[n.length-1];for(let r=0;r<n.length-1;++r)s+=t[r]*n[r];return s}function Xs(n,e,t){if(e===0)return[];if(e===1)return[n];const s=new Array(e);for(let r=0;r<s.length-1;++r)s[r]=Math.floor(n/t[r]),n-=s[r]*t[r];return s[s.length-1]=n,s}function js(n){return n&&n.then&&typeof n.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ir="tfjsflags";class Mc{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Vc,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(y().getBool("IS_TEST")||y().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,s){if(this.flagRegistry[e]={evaluationFn:t,setHook:s},this.urlFlags[e]!=null){const r=this.urlFlags[e];y().getBool("IS_TEST")||y().getBool("PROD")||console.warn(`Setting feature override from URL ${e}: ${r}.`),this.set(e,r)}}async getAsync(e){return e in this.flags?this.flags[e]:(this.flags[e]=await this.evaluateFlag(e),this.flags[e])}get(e){if(e in this.flags)return this.flags[e];const t=this.evaluateFlag(e);if(js(t))throw new Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw new Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw new Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const e=this.getQueryParams(this.global.location.search);Ir in e&&e[Ir].split(",").forEach(s=>{const[r,o]=s.split(":");this.urlFlags[r]=Wc(r,o)})}}function Vc(n){const e={};return n.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(t,...s)=>(Uc(e,s[0],s[1]),s.join("="))),e}function Uc(n,e,t){n[decodeURIComponent(e)]=decodeURIComponent(t||"")}function Wc(n,e){const t=e.toLowerCase();return t==="true"||t==="false"?t==="true":`${+t}`===t?+t:e}function y(){return Ao}let Ao=null;function Gc(n){Ao=n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ds;function Fo(){if(ds==null){let n;if(typeof window<"u")n=window;else if(typeof global<"u")n=global;else if(typeof process<"u")n=process;else if(typeof self<"u")n=self;else throw new Error("Could not find a global object");ds=n}return ds}function zc(){const n=Fo();return n._tfGlobals==null&&(n._tfGlobals=new Map),n._tfGlobals}function qs(n,e){const t=zc();if(t.has(n))return t.get(n);{const s=e();return t.set(n,s),t.get(n)}}const Do="Abs",Hc="Acos",Xc="Acosh",Ks="Add",jc="AddN",qc="All",Kc="Any",Yc="ArgMax",Qc="ArgMin",Zc="Asin",Jc="Asinh",el="Atan",tl="Atanh",nl="Atan2",sl="AvgPool",rl="AvgPoolGrad",ol="AvgPool3D",il="AvgPool3DGrad",al="BatchMatMul",cl="BatchToSpaceND",ll="Bincount",ul="BitwiseAnd",dl="BroadcastArgs",Ys="Cast",hl="Ceil",fl="ClipByValue",Oo="Complex",Po="ComplexAbs",pl="Concat",ml="Conv2D",gl="Conv2DBackpropFilter",xl="Conv2DBackpropInput",Cl="Conv3D",bl="Conv3DBackpropFilterV2",wl="Conv3DBackpropInputV2",yl="Cos",$l="Cosh",vl="Cumprod",Sl="Cumsum",Il="CropAndResize",Rl="DenseBincount",Tl="DepthToSpace",El="DepthwiseConv2dNative",Nl="DepthwiseConv2dNativeBackpropFilter",kl="DepthwiseConv2dNativeBackpropInput",Al="Diag",Fl="Dilation2D",_o="RealDiv",Dl="Einsum",Lo="Elu",Ol="EluGrad",Pl="Erf",_l="Equal",Ll="Exp",Bl="ExpandDims",Ml="Expm1",Vl="FFT",Bo="Fill",Ul="FlipLeftRight",Wl="Floor",Mo="FloorDiv",Gl="FusedBatchNorm",zl="GatherV2",Hl="GatherNd",Xl="Greater",jl="GreaterEqual",Qs="Identity",ql="IFFT",Kl="Imag",Yl="IsFinite",Ql="IsInf",Zl="IsNan",Vo="LeakyRelu",Jl="Less",eu="LessEqual",tu="LinSpace",nu="Log",su="Log1p",ru="LogicalAnd",ou="LogicalNot",iu="LogicalOr",au="LRN",cu="LRNGrad",lu="Max",Uo="Maximum",uu="MaxPool",du="MaxPoolGrad",hu="MaxPool3D",fu="MaxPool3DGrad",pu="MaxPoolWithArgmax",mu="Mean",gu="Min",xu="Minimum",Cu="MirrorPad",bu="Mod",wu="Multinomial",Wo="Multiply",yu="Neg",$u="NotEqual",vu="NonMaxSuppressionV3",Su="NonMaxSuppressionV4",Iu="NonMaxSuppressionV5",Ru="OnesLike",Tu="OneHot",Eu="Pack",Nu="PadV2",Go="Pow",zo="Prelu",ku="Prod",Au="RaggedGather",Fu="RaggedRange",Du="RaggedTensorToTensor",Ou="Range",Pu="Real",_u="Reciprocal",Ho="Relu",Xo="Reshape",Lu="ResizeNearestNeighbor",Bu="ResizeNearestNeighborGrad",Mu="ResizeBilinear",Vu="ResizeBilinearGrad",jo="Relu6",Uu="Reverse",Wu="Round",Gu="Rsqrt",zu="ScatterNd",Hu="TensorScatterUpdate",Xu="SearchSorted",ju="Select",qu="Selu",Ku="Slice",Yu="Sin",Qu="Sinh",Zu="Sign",qo="Sigmoid",Ju="Softplus",Ko="Sqrt",Yo="Sum",ed="SpaceToBatchND",td="SplitV",nd="Softmax",sd="SparseFillEmptyRows",rd="SparseReshape",od="SparseSegmentMean",id="SparseSegmentSum",ad="SparseToDense",cd="SquaredDifference",ld="Square",ud="StaticRegexReplace",dd="StridedSlice",hd="StringNGrams",fd="StringSplit",pd="StringToHashBucketFast",Qo="Sub",md="Tan",gd="Tanh",Zo="Tile",xd="TopK",Cd="Transform",bd="Transpose",wd="Unique",yd="Unpack",$d="UnsortedSegmentSum",Jo="ZerosLike",ei="Step",vd="FromPixels",Sd="RotateWithOffset",Id="_FusedMatMul",Rd="FusedConv2D",Td="FusedDepthwiseConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pe(...n){y().getBool("IS_TEST")||y().getBool("PROD")||console.warn(...n)}function Ed(...n){y().getBool("IS_TEST")||y().getBool("PROD")||console.log(...n)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Un=qs("kernelRegistry",()=>new Map),Nd=qs("gradRegistry",()=>new Map);function Rr(n,e){const t=ti(n,e);return Un.get(t)}function Tr(n){return Nd.get(n)}function Er(n){const e=Un.entries(),t=[];for(;;){const{done:s,value:r}=e.next();if(s)break;const[o,i]=r,[a]=o.split("_");a===n&&t.push(i)}return t}function kd(n){const{kernelName:e,backendName:t}=n,s=ti(e,t);Un.has(s)&&Pe(`The kernel '${e}' for backend '${t}' is already registered`),Un.set(s,n)}function ti(n,e){return`${e}_${n}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ni(n){return n instanceof Float32Array||n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray}var si=G,$e=null;try{$e=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function G(n,e,t){this.low=n|0,this.high=e|0,this.unsigned=!!t}G.prototype.__isLong__;Object.defineProperty(G.prototype,"__isLong__",{value:!0});function xe(n){return(n&&n.__isLong__)===!0}G.isLong=xe;var Nr={},kr={};function $t(n,e){var t,s,r;return e?(n>>>=0,(r=0<=n&&n<256)&&(s=kr[n],s)?s:(t=z(n,(n|0)<0?-1:0,!0),r&&(kr[n]=t),t)):(n|=0,(r=-128<=n&&n<128)&&(s=Nr[n],s)?s:(t=z(n,n<0?-1:0,!1),r&&(Nr[n]=t),t))}G.fromInt=$t;function ve(n,e){if(isNaN(n))return e?ut:Se;if(e){if(n<0)return ut;if(n>=ri)return ai}else{if(n<=-Fr)return me;if(n+1>=Fr)return ii}return n<0?ve(-n,e).neg():z(n%Mt|0,n/Mt|0,e)}G.fromNumber=ve;function z(n,e,t){return new G(n,e,t)}G.fromBits=z;var Wn=Math.pow;function Zs(n,e,t){if(n.length===0)throw Error("empty string");if(n==="NaN"||n==="Infinity"||n==="+Infinity"||n==="-Infinity")return Se;if(typeof e=="number"?(t=e,e=!1):e=!!e,t=t||10,t<2||36<t)throw RangeError("radix");var s;if((s=n.indexOf("-"))>0)throw Error("interior hyphen");if(s===0)return Zs(n.substring(1),e,t).neg();for(var r=ve(Wn(t,8)),o=Se,i=0;i<n.length;i+=8){var a=Math.min(8,n.length-i),c=parseInt(n.substring(i,i+a),t);if(a<8){var l=ve(Wn(t,a));o=o.mul(l).add(ve(c))}else o=o.mul(r),o=o.add(ve(c))}return o.unsigned=e,o}G.fromString=Zs;function Le(n,e){return typeof n=="number"?ve(n,e):typeof n=="string"?Zs(n,e):z(n.low,n.high,typeof e=="boolean"?e:n.unsigned)}G.fromValue=Le;var Ar=65536,Ad=1<<24,Mt=Ar*Ar,ri=Mt*Mt,Fr=ri/2,Dr=$t(Ad),Se=$t(0);G.ZERO=Se;var ut=$t(0,!0);G.UZERO=ut;var Pt=$t(1);G.ONE=Pt;var oi=$t(1,!0);G.UONE=oi;var Rs=$t(-1);G.NEG_ONE=Rs;var ii=z(-1,2147483647,!1);G.MAX_VALUE=ii;var ai=z(-1,-1,!0);G.MAX_UNSIGNED_VALUE=ai;var me=z(0,-2147483648,!1);G.MIN_VALUE=me;var R=G.prototype;R.toInt=function(){return this.unsigned?this.low>>>0:this.low};R.toNumber=function(){return this.unsigned?(this.high>>>0)*Mt+(this.low>>>0):this.high*Mt+(this.low>>>0)};R.toString=function(e){if(e=e||10,e<2||36<e)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(me)){var t=ve(e),s=this.div(t),r=s.mul(t).sub(this);return s.toString(e)+r.toInt().toString(e)}else return"-"+this.neg().toString(e);for(var o=ve(Wn(e,6),this.unsigned),i=this,a="";;){var c=i.div(o),l=i.sub(c.mul(o)).toInt()>>>0,u=l.toString(e);if(i=c,i.isZero())return u+a;for(;u.length<6;)u="0"+u;a=""+u+a}};R.getHighBits=function(){return this.high};R.getHighBitsUnsigned=function(){return this.high>>>0};R.getLowBits=function(){return this.low};R.getLowBitsUnsigned=function(){return this.low>>>0};R.getNumBitsAbs=function(){if(this.isNegative())return this.eq(me)?64:this.neg().getNumBitsAbs();for(var e=this.high!=0?this.high:this.low,t=31;t>0&&!(e&1<<t);t--);return this.high!=0?t+33:t+1};R.isZero=function(){return this.high===0&&this.low===0};R.eqz=R.isZero;R.isNegative=function(){return!this.unsigned&&this.high<0};R.isPositive=function(){return this.unsigned||this.high>=0};R.isOdd=function(){return(this.low&1)===1};R.isEven=function(){return(this.low&1)===0};R.equals=function(e){return xe(e)||(e=Le(e)),this.unsigned!==e.unsigned&&this.high>>>31===1&&e.high>>>31===1?!1:this.high===e.high&&this.low===e.low};R.eq=R.equals;R.notEquals=function(e){return!this.eq(e)};R.neq=R.notEquals;R.ne=R.notEquals;R.lessThan=function(e){return this.comp(e)<0};R.lt=R.lessThan;R.lessThanOrEqual=function(e){return this.comp(e)<=0};R.lte=R.lessThanOrEqual;R.le=R.lessThanOrEqual;R.greaterThan=function(e){return this.comp(e)>0};R.gt=R.greaterThan;R.greaterThanOrEqual=function(e){return this.comp(e)>=0};R.gte=R.greaterThanOrEqual;R.ge=R.greaterThanOrEqual;R.compare=function(e){if(xe(e)||(e=Le(e)),this.eq(e))return 0;var t=this.isNegative(),s=e.isNegative();return t&&!s?-1:!t&&s?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1};R.comp=R.compare;R.negate=function(){return!this.unsigned&&this.eq(me)?me:this.not().add(Pt)};R.neg=R.negate;R.add=function(e){xe(e)||(e=Le(e));var t=this.high>>>16,s=this.high&65535,r=this.low>>>16,o=this.low&65535,i=e.high>>>16,a=e.high&65535,c=e.low>>>16,l=e.low&65535,u=0,d=0,h=0,f=0;return f+=o+l,h+=f>>>16,f&=65535,h+=r+c,d+=h>>>16,h&=65535,d+=s+a,u+=d>>>16,d&=65535,u+=t+i,u&=65535,z(h<<16|f,u<<16|d,this.unsigned)};R.subtract=function(e){return xe(e)||(e=Le(e)),this.add(e.neg())};R.sub=R.subtract;R.multiply=function(e){if(this.isZero())return Se;if(xe(e)||(e=Le(e)),$e){var t=$e.mul(this.low,this.high,e.low,e.high);return z(t,$e.get_high(),this.unsigned)}if(e.isZero())return Se;if(this.eq(me))return e.isOdd()?me:Se;if(e.eq(me))return this.isOdd()?me:Se;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(Dr)&&e.lt(Dr))return ve(this.toNumber()*e.toNumber(),this.unsigned);var s=this.high>>>16,r=this.high&65535,o=this.low>>>16,i=this.low&65535,a=e.high>>>16,c=e.high&65535,l=e.low>>>16,u=e.low&65535,d=0,h=0,f=0,p=0;return p+=i*u,f+=p>>>16,p&=65535,f+=o*u,h+=f>>>16,f&=65535,f+=i*l,h+=f>>>16,f&=65535,h+=r*u,d+=h>>>16,h&=65535,h+=o*l,d+=h>>>16,h&=65535,h+=i*c,d+=h>>>16,h&=65535,d+=s*u+r*l+o*c+i*a,d&=65535,z(f<<16|p,d<<16|h,this.unsigned)};R.mul=R.multiply;R.divide=function(e){if(xe(e)||(e=Le(e)),e.isZero())throw Error("division by zero");if($e){if(!this.unsigned&&this.high===-2147483648&&e.low===-1&&e.high===-1)return this;var t=(this.unsigned?$e.div_u:$e.div_s)(this.low,this.high,e.low,e.high);return z(t,$e.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?ut:Se;var s,r,o;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return ut;if(e.gt(this.shru(1)))return oi;o=ut}else{if(this.eq(me)){if(e.eq(Pt)||e.eq(Rs))return me;if(e.eq(me))return Pt;var i=this.shr(1);return s=i.div(e).shl(1),s.eq(Se)?e.isNegative()?Pt:Rs:(r=this.sub(e.mul(s)),o=s.add(r.div(e)),o)}else if(e.eq(me))return this.unsigned?ut:Se;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();o=Se}for(r=this;r.gte(e);){s=Math.max(1,Math.floor(r.toNumber()/e.toNumber()));for(var a=Math.ceil(Math.log(s)/Math.LN2),c=a<=48?1:Wn(2,a-48),l=ve(s),u=l.mul(e);u.isNegative()||u.gt(r);)s-=c,l=ve(s,this.unsigned),u=l.mul(e);l.isZero()&&(l=Pt),o=o.add(l),r=r.sub(u)}return o};R.div=R.divide;R.modulo=function(e){if(xe(e)||(e=Le(e)),$e){var t=(this.unsigned?$e.rem_u:$e.rem_s)(this.low,this.high,e.low,e.high);return z(t,$e.get_high(),this.unsigned)}return this.sub(this.div(e).mul(e))};R.mod=R.modulo;R.rem=R.modulo;R.not=function(){return z(~this.low,~this.high,this.unsigned)};R.and=function(e){return xe(e)||(e=Le(e)),z(this.low&e.low,this.high&e.high,this.unsigned)};R.or=function(e){return xe(e)||(e=Le(e)),z(this.low|e.low,this.high|e.high,this.unsigned)};R.xor=function(e){return xe(e)||(e=Le(e)),z(this.low^e.low,this.high^e.high,this.unsigned)};R.shiftLeft=function(e){return xe(e)&&(e=e.toInt()),(e&=63)===0?this:e<32?z(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):z(0,this.low<<e-32,this.unsigned)};R.shl=R.shiftLeft;R.shiftRight=function(e){return xe(e)&&(e=e.toInt()),(e&=63)===0?this:e<32?z(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):z(this.high>>e-32,this.high>=0?0:-1,this.unsigned)};R.shr=R.shiftRight;R.shiftRightUnsigned=function(e){if(xe(e)&&(e=e.toInt()),e&=63,e===0)return this;var t=this.high;if(e<32){var s=this.low;return z(s>>>e|t<<32-e,t>>>e,this.unsigned)}else return e===32?z(t,0,this.unsigned):z(t>>>e-32,0,this.unsigned)};R.shru=R.shiftRightUnsigned;R.shr_u=R.shiftRightUnsigned;R.toSigned=function(){return this.unsigned?z(this.low,this.high,!1):this};R.toUnsigned=function(){return this.unsigned?this:z(this.low,this.high,!0)};R.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()};R.toBytesLE=function(){var e=this.high,t=this.low;return[t&255,t>>>8&255,t>>>16&255,t>>>24,e&255,e>>>8&255,e>>>16&255,e>>>24]};R.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,e&255,t>>>24,t>>>16&255,t>>>8&255,t&255]};G.fromBytes=function(e,t,s){return s?G.fromBytesLE(e,t):G.fromBytesBE(e,t)};G.fromBytesLE=function(e,t){return new G(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)};G.fromBytesBE=function(e,t){return new G(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)};const ci=Ic(si),Fd=Rc({__proto__:null,default:ci},[si]);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const at=ci||Fd;function Jn(n){return at.fromString(n,!0,16)}const li=Jn("c3a5c85c97cb3127"),it=Jn("b492b66fbe98f273"),oe=Jn("9ae16a3b2f90404f");function Ts(n){return n.xor(n.shru(47))}function ui(n,e,t){const s=n.slice(e,e+t);return at.fromBytes(Array.from(s),!0,!0)}function W(n,e){return ui(n,e,8)}function Or(n,e){return ui(n,e,4)}function j(n,e){return e===0?n:n.shru(e).or(n.shl(64-e))}function tt(n,e,t=Jn("9ddfea08eb382d69")){let s=n.xor(e).mul(t);s=s.xor(s.shru(47));let r=e.xor(s).mul(t);return r=r.xor(r.shru(47)),r=r.mul(t),r}function Dd(n,e,t,s,r,o){r=r.add(n),o=j(o.add(r).add(s),21);const i=r;return r=r.add(e),r=r.add(t),o=o.add(j(r,44)),[r.add(s),o.add(i)]}function Tn(n,e,t,s){return Dd(W(n,e),W(n,e+8),W(n,e+16),W(n,e+24),t,s)}function Od(n,e=n.length){if(e>=8){const t=oe.add(e*2),s=W(n,0).add(oe),r=W(n,e-8),o=j(r,37).mul(t).add(s),i=j(s,25).add(r).mul(t);return tt(o,i,t)}if(e>=4){const t=oe.add(e*2),s=Or(n,0);return tt(s.shl(3).add(e),Or(n,e-4),t)}if(e>0){const t=n[0],s=n[e>>1],r=n[e-1],o=t+(s<<8),i=e+(r<<2);return Ts(oe.mul(o).xor(li.mul(i))).mul(oe)}return oe}function Pd(n,e=n.length){const t=oe.add(e*2),s=W(n,0).mul(it),r=W(n,8),o=W(n,e-8).mul(t),i=W(n,e-16).mul(oe);return tt(j(s.add(r),43).add(j(o,30)).add(i),s.add(j(r.add(oe),18)).add(o),t)}function _d(n,e=n.length){const t=oe.add(e*2),s=W(n,0).mul(oe),r=W(n,8),o=W(n,e-8).mul(t),i=W(n,e-16).mul(oe),a=j(s.add(r),43).add(j(o,30)).add(i),c=tt(a,s.add(j(r.add(oe),18)).add(o),t),l=W(n,16).mul(t),u=W(n,24),d=a.add(W(n,e-32)).mul(t),h=c.add(W(n,e-24)).mul(t);return tt(j(l.add(u),43).add(j(d,30)).add(h),l.add(j(u.add(s),18)).add(d),t)}function Ld(n,e=n.length){const t=at.fromNumber(81,!0);if(e<=32)return e<=16?Od(n,e):Pd(n,e);if(e<=64)return _d(n,e);let s=t,r=t.mul(it).add(113),o=Ts(r.mul(oe).add(113)).mul(oe),i=[at.UZERO,at.UZERO],a=[at.UZERO,at.UZERO];s=s.mul(oe).add(W(n,0));let c=0;const l=(e-1>>6)*64,u=l+(e-1&63)-63;do s=j(s.add(r).add(i[0]).add(W(n,c+8)),37).mul(it),r=j(r.add(i[1]).add(W(n,c+48)),42).mul(it),s=s.xor(a[1]),r=r.add(i[0]).add(W(n,c+40)),o=j(o.add(a[0]),33).mul(it),i=Tn(n,c,i[1].mul(it),s.add(a[0])),a=Tn(n,c+32,o.add(a[1]),r.add(W(n,c+16))),[o,s]=[s,o],c+=64;while(c!==l);const d=it.add(o.and(255).shl(1));return c=u,a[0]=a[0].add(e-1&63),i[0]=i[0].add(a[0]),a[0]=a[0].add(i[0]),s=j(s.add(r).add(i[0]).add(W(n,c+8)),37).mul(d),r=j(r.add(i[1]).add(W(n,c+48)),42).mul(d),s=s.xor(a[1].mul(9)),r=r.add(i[0].mul(9).add(W(n,c+40))),o=j(o.add(a[0]),33).mul(d),i=Tn(n,c,i[1].mul(d),s.add(a[0])),a=Tn(n,c+32,o.add(a[1]),r.add(W(n,c+16))),[o,s]=[s,o],tt(tt(i[0],a[0],d).add(Ts(r).mul(li)).add(o),tt(i[1],a[1],d).add(s),d)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xt(n,e){return e==="string"?ht(n):es([n],e)}function Bd(n,e){return n instanceof Float32Array&&e==="float32"||n instanceof Int32Array&&e==="int32"||n instanceof Uint8Array&&e==="bool"}function es(n,e){if(e==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(n)&&(n=mt(n)),y().getBool("DEBUG")&&Fc(n,e),Bd(n,e))return n;if(e==null||e==="float32"||e==="complex64")return new Float32Array(n);if(e==="int32")return new Int32Array(n);if(e==="bool"){const t=new Uint8Array(n.length);for(let s=0;s<t.length;++s)Math.round(n[s])!==0&&(t[s]=1);return t}else throw new Error(`Unknown data type ${e}`)}function Ae(){return y().platform.now()}function ht(n,e="utf-8"){return e=e||"utf-8",y().platform.encode(n,e)}function Vt(n,e="utf-8"){return e=e||"utf-8",y().platform.decode(n,e)}function Re(n){return y().platform.isTypedArray!=null?y().platform.isTypedArray(n):ni(n)}function mt(n,e=[],t=!1){if(e==null&&(e=[]),typeof n=="boolean"||typeof n=="number"||typeof n=="string"||js(n)||n==null||Re(n)&&t)e.push(n);else if(Array.isArray(n)||Re(n))for(let s=0;s<n.length;++s)mt(n[s],e,t);else{let s=-1;for(const r of Object.keys(n))/^([1-9]+[0-9]*|0)$/.test(r)&&(s=Math.max(s,Number(r)));for(let r=0;r<=s;r++)mt(n[r],e,t)}return e}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Md{constructor(e,t){this.backendTimer=e,this.logger=t,t==null&&(this.logger=new Ud)}profileKernel(e,t,s){let r;const o=()=>{r=s()};let i;const a=Ae();if(this.backendTimer.timerAvailable())i=this.backendTimer.time(o);else{o();for(const l of r)l.dataSync();i=Promise.resolve({kernelMs:Ae()-a})}if(y().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let l=0;l<r.length;l++){const u=r[l];u.data().then(d=>{Vd(d,u.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:i.then(l=>l.kernelMs),extraInfo:i.then(l=>l.getExtraProfileInfo!=null?l.getExtraProfileInfo():"")}}logKernelProfile(e){const{kernelName:t,outputs:s,timeMs:r,inputs:o,extraInfo:i}=e;s.forEach(a=>{Promise.all([a.data(),r,i]).then(c=>{this.logger.logKernelProfile(t,a,c[0],c[1],o,c[2])})})}}function Vd(n,e,t){if(e!=="float32")return!1;for(let s=0;s<n.length;s++){const r=n[s];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${t}'`),!0}return!1}class Ud{logKernelProfile(e,t,s,r,o,i){const a=typeof r=="number"?_t(`${r}ms`,9):r.error,c=_t(e,25),l=t.rank,u=t.size,d=_t(t.shape.toString(),14);let h="";for(const f in o){const p=o[f];if(p!=null){const x=p.shape||t.shape,g=x.length;h+=`${f}: ${g}D ${g>0?x:""} `}}console.log(`%c${c}	%c${a}	%c${l}D ${d}	%c${u}	%c${h}	%c${i}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wd(n,e,t){const s={},r={};for(let c=0;c<e.length;c++)s[e[c].id]=!0;for(let c=0;c<n.length;c++){const l=n[c],u=l.inputs;for(const d in u){const h=u[d];let f=!1;for(let p=0;p<e.length;p++)if(s[h.id]){l.outputs.forEach(x=>s[x.id]=!0),f=!0,r[l.id]=!0;break}if(f)break}}const o={};o[t.id]=!0;const i={};for(let c=n.length-1;c>=0;c--){const l=n[c],u=l.inputs;for(let d=0;d<l.outputs.length;d++)if(o[l.outputs[d].id]){for(const h in u)o[u[h].id]=!0,i[l.id]=!0;break}}const a=[];for(let c=0;c<n.length;c++){const l=n[c];if(r[l.id]&&i[l.id]){const u={};for(const h in l.inputs){const f=l.inputs[h];s[f.id]&&(u[h]=f)}const d=Object.assign({},l);d.inputs=u,d.outputs=l.outputs,a.push(d)}}return a}function Gd(n,e,t,s){for(let r=e.length-1;r>=0;r--){const o=e[r],i=[];if(o.outputs.forEach(c=>{const l=n[c.id];l!=null?i.push(l):i.push(null)}),o.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${o.kernelName}.`);const a=o.gradient(i);for(const c in o.inputs){if(!(c in a))throw new Error(`Cannot backprop through input ${c}. Available gradients found: ${Object.keys(a)}.`);const l=t(()=>a[c]());if(l.dtype!=="float32")throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input ${c} must have 'float32' dtype, but has '${l.dtype}'`);const u=o.inputs[c];if(!J(l.shape,u.shape))throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input '${c}' has shape '${l.shape}', which does not match the shape of the input '${u.shape}'`);if(n[u.id]==null)n[u.id]=l;else{const d=n[u.id];n[u.id]=s(d,l),d.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pr=20,on=3,hs=7;function zd(n,e,t,s){const r=Z(e),o=Hd(n,e,t,r),i=e.length,a=_n(n,e,t,r,o),c=["Tensor"];return s&&(c.push(`  dtype: ${t}`),c.push(`  rank: ${i}`),c.push(`  shape: [${e}]`),c.push("  values:")),c.push(a.map(l=>"    "+l).join(`
`)),c.join(`
`)}function Hd(n,e,t,s){const r=E(e),o=s[s.length-1],i=new Array(o).fill(0),a=e.length,c=t==="complex64"?cn(n):n;if(a>1)for(let l=0;l<r/o;l++){const u=l*o;for(let d=0;d<o;d++)i[d]=Math.max(i[d],an(c[u+d],0,t).length)}return i}function an(n,e,t){let s;return Array.isArray(n)?s=`${parseFloat(n[0].toFixed(hs))} + ${parseFloat(n[1].toFixed(hs))}j`:Zn(n)?s=`'${n}'`:t==="bool"?s=di(n):s=parseFloat(n.toFixed(hs)).toString(),_t(s,e)}function di(n){return n===0?"false":"true"}function _n(n,e,t,s,r,o=!0){const i=t==="complex64"?2:1,a=e[0],c=e.length;if(c===0){if(t==="complex64"){const x=cn(n);return[an(x[0],0,t)]}return t==="bool"?[di(n[0])]:[n[0].toString()]}if(c===1){if(a>Pr){const g=on*i;let m=Array.from(n.slice(0,g)),C=Array.from(n.slice((a-on)*i,a*i));return t==="complex64"&&(m=cn(m),C=cn(C)),["["+m.map((b,w)=>an(b,r[w],t)).join(", ")+", ..., "+C.map((b,w)=>an(b,r[a-on+w],t)).join(", ")+"]"]}return["["+(t==="complex64"?cn(n):Array.from(n)).map((g,m)=>an(g,r[m],t)).join(", ")+"]"]}const l=e.slice(1),u=s.slice(1),d=s[0]*i,h=[];if(a>Pr){for(let x=0;x<on;x++){const g=x*d,m=g+d;h.push(..._n(n.slice(g,m),l,t,u,r,!1))}h.push("...");for(let x=a-on;x<a;x++){const g=x*d,m=g+d;h.push(..._n(n.slice(g,m),l,t,u,r,x===a-1))}}else for(let x=0;x<a;x++){const g=x*d,m=g+d;h.push(..._n(n.slice(g,m),l,t,u,r,x===a-1))}const f=c===2?",":"";h[0]="["+(a>0?h[0]+f:"");for(let x=1;x<h.length-1;x++)h[x]=" "+h[x]+f;let p=`,
`;for(let x=2;x<c;x++)p+=`
`;return h[h.length-1]=" "+h[h.length-1]+"]"+(o?"":p),h}function cn(n){const e=[];for(let t=0;t<n.length;t+=2)e.push([n[t],n[t+1]]);return e}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Gn{constructor(e,t,s){if(this.dtype=t,this.shape=e.slice(),this.size=E(e),s!=null){const r=s.length;I(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(t==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||q(t,this.size),this.strides=Z(e)}set(e,...t){t.length===0&&(t=[0]),I(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);const s=this.locToIndex(t);this.values[s]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(const r of e){if(r<0||r>=this.shape[t]){const o=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw new Error(o)}t++}let s=e[e.length-1];for(let r=0;r<e.length-1;++r)s+=this.strides[r]*e[r];return this.values[s]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let s=0;s<e.length-1;++s)t+=this.strides[s]*e[s];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];const t=new Array(this.shape.length);for(let s=0;s<t.length-1;++s)t[s]=Math.floor(e/this.strides[s]),e-=t[s]*this.strides[s];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return Fe().makeTensor(this.values,this.shape,this.dtype)}}let Fe=null,Dt=null;function Xd(n){Fe=n}function jd(n){Dt=n}class De{constructor(e,t,s,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||"float32",this.size=E(e),this.strides=Z(e),this.dataId=s,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const e=await this.data();return Dt.buffer(this.shape,this.dtype,e)}bufferSync(){return Dt.buffer(this.shape,this.dtype,this.dataSync())}async array(){const e=await this.data();return Sr(this.shape,e,this.dtype==="complex64")}arraySync(){return Sr(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const e=Fe().read(this.dataId);if(this.dtype==="string"){const t=await e;try{return t.map(s=>Vt(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),Fe().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();const e=Fe().readSync(this.dataId);if(this.dtype==="string")try{return e.map(t=>Vt(t))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();const e=await Fe().read(this.dataId);return this.dtype==="string"?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),Fe().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(e=!1){return Dt.print(this,e)}clone(){return this.throwIfDisposed(),Dt.clone(this)}toString(e=!1){const t=this.dataSync();return zd(t,this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Dt.cast(this,e)}variable(e=!0,t,s){return this.throwIfDisposed(),Fe().makeVariable(this,e,t,s)}}Object.defineProperty(De,Symbol.hasInstance,{value:n=>!!n&&n.data!=null&&n.dataSync!=null&&n.throwIfDisposed!=null});function hi(){return qs("Tensor",()=>De)}hi();class zn extends De{constructor(e,t,s,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=s}assign(e){if(e.dtype!==this.dtype)throw new Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!J(e.shape,this.shape))throw new Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);Fe().disposeTensor(this),this.dataId=e.dataId,Fe().incRef(this,null)}dispose(){Fe().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(zn,Symbol.hasInstance,{value:n=>n instanceof De&&n.assign!=null&&n.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var _r;(function(n){n.R0="R0",n.R1="R1",n.R2="R2",n.R3="R3",n.R4="R4",n.R5="R5",n.R6="R6"})(_r||(_r={}));var Es;(function(n){n.float32="float32",n.int32="int32",n.bool="int32",n.complex64="complex64"})(Es||(Es={}));var Ns;(function(n){n.float32="float32",n.int32="int32",n.bool="bool",n.complex64="complex64"})(Ns||(Ns={}));var ks;(function(n){n.float32="float32",n.int32="float32",n.bool="float32",n.complex64="complex64"})(ks||(ks={}));var As;(function(n){n.float32="complex64",n.int32="complex64",n.bool="complex64",n.complex64="complex64"})(As||(As={}));const qd={float32:ks,int32:Es,bool:Ns,complex64:As};function ze(n,e){if(n==="string"||e==="string"){if(n==="string"&&e==="string")return"string";throw new Error(`Can not upcast ${n} with ${e}`)}return qd[n][e]}function Js(n){return ze(n,"int32")}function fi(n){return n!=null&&typeof n=="object"&&"texture"in n&&n.texture instanceof WebGLTexture}function pi(n){return typeof GPUBuffer<"u"&&n!=null&&typeof n=="object"&&"buffer"in n&&n.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vt(n,e){if(n.dtype===e.dtype)return[n,e];const t=ze(n.dtype,e.dtype);return[n.cast(t),e.cast(t)]}function mi(n){const e=[];return gi(n,e,new Set),e}function gi(n,e,t){if(n==null)return;if(n instanceof De){e.push(n);return}if(!Kd(n))return;const s=n;for(const r in s){const o=s[r];t.has(o)||(t.add(o),gi(o,e,t))}}function Kd(n){return Array.isArray(n)||typeof n=="object"}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fs(n){return n.kernelName!=null}class Lr{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(const e in this.registeredVariables)this.registeredVariables[e].dispose()}}class Ut{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Lr}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const s=e[t];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw new Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry))if(e in this.registryFactory){const{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null;return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,s=1){return e in this.registryFactory?(Pe(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:s},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw new Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;const{success:t,asyncInit:s}=this.initializeBackend(e);if(!(s?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new Md(this.backendInstance),!0}setupRegisteredKernels(){Er(this.backendName).forEach(t=>{t.setupFunc!=null&&t.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){Er(e).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[e])})}initializeBackend(e){const t=this.registryFactory[e];if(t==null)throw new Error(`Cannot initialize backend ${e}, no registration found.`);try{const s=t.factory();if(s&&!(s instanceof Eo)&&typeof s.then=="function"){const r=++this.pendingBackendInitId,o=s.then(i=>r<this.pendingBackendInitId?!1:(this.registry[e]=i,this.pendingBackendInit=null,!0)).catch(i=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,Pe(`Initialization of backend ${e} failed`),Pe(i.stack||i.message)),!1));return this.pendingBackendInit=o,{success:o,asyncInit:!0}}else return this.registry[e]=s,{success:!0,asyncInit:!1}}catch(s){return Pe(`Initialization of backend ${e} failed`),Pe(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw new Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const s=e[t],{success:r,asyncInit:o}=this.initializeBackend(s);if(o||r)return{name:s,asyncInit:o}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,t){const s=this.state.tensorInfo.get(t),r=s.backend,o=this.readSync(t),i=r.refCount(t);r.disposeData(t,!0),s.backend=e,e.move(t,o,s.shape,s.dtype,i),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let s=null;if(t==null){if(typeof e!="function")throw new Error("Please provide a function to tidy()");t=e}else{if(typeof e!="string"&&!(e instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof t!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=e}let r;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(e,t,s){e();try{const r=s();return t(),r}catch(r){throw t(),r}}nextTensorId(){return Ut.nextTensorId++}nextVariableId(){return Ut.nextVariableId++}clone(e){const t=F.runKernel(Qs,{x:e}),s={x:e},r=i=>({x:()=>{const a="float32",c={x:i},l={dtype:a};return F.runKernel(Ys,c,l)}}),o=[];return this.addTapeNode(this.state.activeScope.name,s,[t],r,o,{}),t}runKernel(e,t,s){if(this.backendName==null&&this.backend,!(Rr(e,this.backendName)!=null))throw new Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,t,s){const r=this.backend.numDataIds();let o=0;s.forEach(c=>{o+=c.dtype==="complex64"?3:1});const i=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=r-t-o-i;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${e}'`)}runKernelFunc(e){let t,s=[];const r=this.isTapeOn(),o=this.state.numBytes,i=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let a;this.backendName==null&&this.backend;let c;const l=fs(e)?e.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(fs(e)){const{kernelName:p,inputs:x,attrs:g}=e;this.backendName==null&&this.backend;const m=Rr(p,this.backendName);I(m!=null,()=>`Cannot find registered kernel '${p}' for backend '${this.backendName}'`),a=()=>{const C=this.backend.numDataIds();c=m.kernelFunc({inputs:x,attrs:g,backend:this.backend});const b=Array.isArray(c)?c:[c];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(p,C,b);const w=b.map($=>$.rank!=null?$:this.makeTensorFromTensorInfo($));if(r){const $=this.getTensorsForGradient(p,x,w);s=this.saveTensorsForBackwardMode($)}return w}}else{const{forwardFunc:p}=e,x=g=>{r&&(s=g.map(m=>this.keep(this.clone(m))))};a=()=>{const g=this.backend.numDataIds();c=this.tidy(()=>p(this.backend,x));const m=Array.isArray(c)?c:[c];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,g,m),m}}const{inputs:u,attrs:d}=e,h=fs(e)?null:e.backwardsFunc;let f;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?t=a():(f=this.profiler.profileKernel(l,u,()=>a()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(f),t=f.outputs)}),r&&this.addTapeNode(l,u,t,h,s,d),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-o,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-i,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(u).map(p=>u[p]!=null?u[p].shape:null),outputShapes:t.map(p=>p.shape),kernelTimeMs:f.timeMs,extraInfo:f.extraInfo}),Array.isArray(c)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(e,t,s){const r=Tr(e);if(r!=null){const o=r.inputsToSave||[],i=r.outputsToSave||[];let a;r.saveAllInputs?(I(Array.isArray(t),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(t).map(l=>t[l])):a=o.map(l=>t[l]);const c=s.filter((l,u)=>i[u]);return a.concat(c)}return[]}makeTensor(e,t,s,r){if(e==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",r=r||this.backend;let o=e;s==="string"&&Zn(e[0])&&(o=e.map(c=>ht(c)));const i=r.write(o,t,s),a=new De(t,s,i,this.nextTensorId());if(this.trackTensor(a,r),s==="string"){const c=this.state.tensorInfo.get(i),l=Pc(o);this.state.numBytes+=l-c.bytes,c.bytes=l}return a}makeTensorFromDataId(e,t,s,r){s=s||"float32";const o={dataId:e,shape:t,dtype:s};return this.makeTensorFromTensorInfo(o,r)}makeTensorFromTensorInfo(e,t){const{dataId:s,shape:r,dtype:o}=e,i=new De(r,o,s,this.nextTensorId());return this.trackTensor(i,t),i}makeVariable(e,t=!0,s,r){s=s||this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));const o=new zn(e,t,s,this.nextTensorId());if(this.state.registeredVariables[o.name]!=null)throw new Error(`Variable with name ${o.name} was already registered`);return this.state.registeredVariables[o.name]=o,this.incRef(o,this.backend),o}trackTensor(e,t){this.state.numTensors++,e.dtype==="string"&&this.state.numStringTensors++;let s=0;e.dtype!=="complex64"&&e.dtype!=="string"&&(s=e.size*Vn(e.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:s})),e instanceof zn||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;const t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!=="complex64"&&e.dtype!=="string"){const s=e.size*Vn(e.dtype);this.state.numBytes-=s}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(const e in this.state.registeredVariables){const t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){const e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons==null&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;const t=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,s,r,o,i){const a={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:s,saved:o},c=Tr(e);c!=null&&(r=c.gradFunc),r!=null&&(a.gradient=l=>(l=l.map((u,d)=>{if(u==null){const h=s[d],f=nt(h.size,h.dtype);return this.makeTensor(f,h.shape,h.dtype)}return u}),r(l.length>1?l:l[0],o,i))),this.state.activeTape.push(a)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){const t={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){const t=mi(e),s=new Set(t.map(o=>o.id));for(let o=0;o<this.state.activeScope.track.length;o++){const i=this.state.activeScope.track[o];!i.kept&&!s.has(i.id)&&i.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(o=>{!o.kept&&o.scopeId===r.id&&this.track(o)})}gradients(e,t,s,r=!1){if(I(t.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const o=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));I(o instanceof De,()=>"The result y returned by f() must be a tensor.");const i=Wd(this.state.activeTape,t,o);if(!r&&i.length===0&&t.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const a={};a[o.id]=s??Yd(o.shape),Gd(a,i,l=>this.tidy(l),Qd);const c=t.map(l=>a[l.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(l=>{for(const u of l.saved)u.dispose()}),this.state.activeTape=null),{value:o,grads:c}})}customGrad(e){return I(vs(e),()=>"The f passed in customGrad(f) must be a function."),(...t)=>{I(t.every(a=>a instanceof De),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const r={};t.forEach((a,c)=>{r[c]=a});const o=(a,c)=>(s=e(...t,c),I(s.value instanceof De,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),I(vs(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),i=(a,c)=>{const l=s.gradFunc(a,c),u=Array.isArray(l)?l:[l];I(u.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),I(u.every(h=>h instanceof De),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const d={};return u.forEach((h,f)=>{d[f]=()=>h}),d};return this.runKernelFunc({forwardFunc:o,backwardsFunc:i,inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){const t=Ae(),s=await this.backend.time(e);return s.wallMs=Ae()-t,s}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Lr;for(const e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}Ut.nextTensorId=0;Ut.nextVariableId=0;function Yd(n){const e=Bc(E(n),"float32");return F.makeTensor(e,n,"float32")}function xi(){const n=Fo();if(n._tfengine==null){const e=new Mc(n);n._tfengine=new Ut(e)}return Gc(n._tfengine.ENV),Xd(()=>n._tfengine),n._tfengine}const F=xi();function Qd(n,e){const t={a:n,b:e};return F.runKernel(Ks,t)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zd(){return typeof navigator<"u"&&navigator!=null}function Ci(n){if(n||Zd()){if(n||(n=navigator),n.product==="ReactNative")return!0;const e=n.userAgent||n.vendor||(typeof window<"u"?window.opera:"");if(!e){const t=n;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(e)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(e.substr(0,4))}return!1}function bi(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ue=y();ue.registerFlag("DEBUG",()=>!1,n=>{n&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});ue.registerFlag("IS_BROWSER",()=>bi());ue.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");ue.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));ue.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));ue.registerFlag("PROD",()=>!1);ue.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>ue.getBool("DEBUG"));ue.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);ue.registerFlag("IS_TEST",()=>!1);ue.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>ue.getBool("DEBUG"));ue.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);ue.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);ue.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jd(n,e){let t=n;if(Re(n))return e==="string"?[]:[n.length];if(fi(n)){const r=n.channels||"RGBA";return[n.height,n.width*r.length]}else if(pi(n))return[n.buffer.size/(e==null?4:Vn(e))];if(!Array.isArray(n))return[];const s=[];for(;Array.isArray(t)||Re(t)&&e!=="string";)s.push(t.length),t=t[0];return Array.isArray(n)&&y().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&wi(n,s,[]),s}function wi(n,e,t){if(t=t||[],!Array.isArray(n)&&!Re(n)){I(e.length===0,()=>`Element arr[${t.join("][")}] is a primitive, but should be an array/TypedArray of ${e[0]} elements`);return}I(e.length>0,()=>`Element arr[${t.join("][")}] should be a primitive, but is an array of ${n.length} elements`),I(n.length===e[0],()=>`Element arr[${t.join("][")}] should have ${e[0]} elements, but has ${n.length} elements`);const s=e.slice(1);for(let r=0;r<n.length;++r)wi(n[r],s,t.concat(r))}function Br(n,e,t,s){if(n!=="string_or_numeric"){if(n==null)throw new Error("Expected dtype cannot be null.");if(n!=="numeric"&&n!==e||n==="numeric"&&e==="string")throw new Error(`Argument '${t}' passed to '${s}' must be ${n} tensor, but got ${e} tensor`)}}function B(n,e,t,s="numeric"){if(n instanceof hi())return Br(s,n.dtype,e,t),n;let r=xn(n);if(r!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(r=s),Br(s,r,e,t),n==null||!Re(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string"){const c=n==null?"null":n.constructor.name;throw new Error(`Argument '${e}' passed to '${t}' must be a Tensor or TensorLike, but got '${c}'`)}const o=Jd(n,r);!Re(n)&&!Array.isArray(n)&&(n=[n]);const a=r!=="string"?es(n,r):mt(n,[],!0);return F.makeTensor(a,o,r)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const eh="__op";function H(n){const e=Object.keys(n);if(e.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${e.length} keys.`);let t=e[0];const s=n[t];t.endsWith("_")&&(t=t.substring(0,t.length-1)),t=t+eh;const r=(...o)=>{F.startScope(t);try{const i=s(...o);return js(i)&&console.error("Cannot return a Promise inside of tidy."),F.endScope(i),i}catch(i){throw F.endScope(null),i}};return Object.defineProperty(r,"name",{value:t,configurable:!0}),r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function th(n,e){const t=B(n,"real","complex"),s=B(e,"imag","complex");No(t.shape,s.shape,`real and imag shapes, ${t.shape} and ${s.shape}, must match in call to tf.complex().`);const r={real:t,imag:s};return F.runKernel(Oo,r)}const nh=H({complex_:th});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sh(n,e,t,s){if(s==null)s=xn(n);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(pi(n)||fi(n)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return F.backend.createTensorFromGPUData(n,e||t,s)}if(!Re(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(e!=null){Cn(e);const r=E(e),o=E(t);I(r===o,()=>`Based on the provided shape, [${e}], the tensor should have ${r} values but has ${o}`);for(let i=0;i<t.length;++i){const a=t[i],c=i===t.length-1?a!==E(e.slice(i)):!0;I(t[i]===e[i]||!c,()=>`Error creating a new Tensor. Inferred shape (${t}) does not match the provided shape (${e}). `)}}return!Re(n)&&!Array.isArray(n)&&(n=[n]),e=e||t,n=s!=="string"?es(n,s):mt(n,[],!0),F.makeTensor(n,e,s)}class St{static join(e){return new St(e).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(s=>Re(s)?s.buffer:s),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let s=0;s<e.length;s++){const r=e[s];s!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const o=t+r.byteLength;this.shards.push({buffer:r,start:t,end:o}),t=o}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e)return new ArrayBuffer(0);const s=this.findShardForByte(e);if(s===-1)throw new Error(`Could not find start shard for byte ${e}`);const r=t-e,o=new ArrayBuffer(r),i=new Uint8Array(o);let a=0;for(let c=s;c<this.shards.length;c++){const l=this.shards[c],d=e+a-l.start,h=a,p=Math.min(t,l.end)-l.start,x=new Uint8Array(l.buffer,d,p-d);if(i.set(x,h),a+=x.length,t<l.end)break}return o}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(r){return e<r.start?-1:e>=r.end?1:0}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=rh(this.shards,t);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function rh(n,e){let t=0,s=n.length;for(;t<=s;){const r=Math.floor((s-t)/2)+t,o=e(n[r]);if(o===0)return r;o<0?s=r:t=r+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qe(){return F}function X(n,e){return F.tidy(n,e)}function be(n){mi(n).forEach(t=>t.dispose())}function oh(n){return F.keep(n)}function pI(n){return F.setBackend(n)}function mI(){return F.ready()}function ih(n,e,t=1){return F.registerBackend(n,e,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const er=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function Mr(n){return er?Buffer.byteLength(n,"utf8"):new Blob([n]).size}function ah(n){if(er)return Buffer.from(n).toString("base64");const e=new Uint8Array(n);let t="";for(let s=0,r=e.length;s<r;s++)t+=String.fromCharCode(e[s]);return btoa(t)}function ch(n){if(er){const s=Buffer.from(n,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const e=atob(n),t=new Uint8Array(e.length);for(let s=0;s<e.length;++s)t.set([e.charCodeAt(s)],s);return t.buffer}function yi(n,e){const t={modelTopology:n.modelTopology,format:n.format,generatedBy:n.generatedBy,convertedBy:n.convertedBy,weightsManifest:e};return n.signature!=null&&(t.signature=n.signature),n.userDefinedMetadata!=null&&(t.userDefinedMetadata=n.userDefinedMetadata),n.modelInitializer!=null&&(t.modelInitializer=n.modelInitializer),n.initializerSignature!=null&&(t.initializerSignature=n.initializerSignature),n.trainingConfig!=null&&(t.trainingConfig=n.trainingConfig),t}function lh(n,e,t){const s={modelTopology:n.modelTopology,format:n.format,generatedBy:n.generatedBy,convertedBy:n.convertedBy};if(n.trainingConfig!=null&&(s.trainingConfig=n.trainingConfig),n.weightsManifest!=null){if(!e)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!t)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=e,s.weightData=t}return n.signature!=null&&(s.signature=n.signature),n.userDefinedMetadata!=null&&(s.userDefinedMetadata=n.userDefinedMetadata),n.modelInitializer!=null&&(s.modelInitializer=n.modelInitializer),n.initializerSignature!=null&&(s.initializerSignature=n.initializerSignature),s}async function uh(n,e){let t,s;return n.weightsManifest!=null&&([t,s]=await e(n.weightsManifest)),lh(n,t,s)}function ts(n){if(n.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:n.modelTopology==null?0:Mr(JSON.stringify(n.modelTopology)),weightSpecsBytes:n.weightSpecs==null?0:Mr(JSON.stringify(n.weightSpecs)),weightDataBytes:n.weightData==null?0:new St(n.weightData).byteLength}}function Vr(n){const e=[];for(const t of n)e.push(...t.weights);return e}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Y{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return Y.instance==null&&(Y.instance=new Y),Y.instance}static registerSaveRouter(e){Y.getInstance().saveRouters.push(e)}static registerLoadRouter(e){Y.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return Y.getHandlers(e,"save")}static getLoadHandlers(e,t){return Y.getHandlers(e,"load",t)}static getHandlers(e,t,s){const r=[];return(t==="load"?Y.getInstance().loadRouters:Y.getInstance().saveRouters).forEach(i=>{const a=i(e,s);a!==null&&r.push(a)}),r}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fs="tensorflowjs",Ds=1,dt="models_store",Je="model_info_store";function $i(){if(!y().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const n=typeof window>"u"?self:window,e=n.indexedDB||n.mozIndexedDB||n.webkitIndexedDB||n.msIndexedDB||n.shimIndexedDB;if(e==null)throw new Error("The current browser does not appear to support IndexedDB.");return e}function Os(n){const e=n.result;e.createObjectStore(dt,{keyPath:"modelPath"}),e.createObjectStore(Je,{keyPath:"modelPath"})}class gt{constructor(e){if(this.indexedDB=$i(),e==null||!e)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((s,r)=>{const o=this.indexedDB.open(Fs,Ds);o.onupgradeneeded=()=>Os(o),o.onsuccess=()=>{const i=o.result;if(t==null){const a=i.transaction(dt,"readonly"),l=a.objectStore(dt).get(this.modelPath);l.onsuccess=()=>{if(l.result==null)return i.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(l.result.modelArtifacts)},l.onerror=u=>(i.close(),r(l.error)),a.oncomplete=()=>i.close()}else{t.weightData=St.join(t.weightData);const a=ts(t),c=i.transaction(Je,"readwrite");let l=c.objectStore(Je),u;try{u=l.put({modelPath:this.modelPath,modelArtifactsInfo:a})}catch(h){return r(h)}let d;u.onsuccess=()=>{d=i.transaction(dt,"readwrite");const h=d.objectStore(dt);let f;try{f=h.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:a})}catch(p){return r(p)}f.onsuccess=()=>s({modelArtifactsInfo:a}),f.onerror=p=>{l=c.objectStore(Je);const x=l.delete(this.modelPath);x.onsuccess=()=>(i.close(),r(f.error)),x.onerror=g=>(i.close(),r(f.error))}},u.onerror=h=>(i.close(),r(u.error)),c.oncomplete=()=>{d==null?i.close():d.oncomplete=()=>i.close()}}},o.onerror=i=>r(o.error)})}}gt.URL_SCHEME="indexeddb://";const vi=n=>y().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(gt.URL_SCHEME)?dh(n.slice(gt.URL_SCHEME.length)):null;Y.registerSaveRouter(vi);Y.registerLoadRouter(vi);function dh(n){return new gt(n)}function hh(n){return n.startsWith(gt.URL_SCHEME)?n.slice(gt.URL_SCHEME.length):n}class fh{constructor(){this.indexedDB=$i()}async listModels(){return new Promise((e,t)=>{const s=this.indexedDB.open(Fs,Ds);s.onupgradeneeded=()=>Os(s),s.onsuccess=()=>{const r=s.result,o=r.transaction(Je,"readonly"),a=o.objectStore(Je).getAll();a.onsuccess=()=>{const c={};for(const l of a.result)c[l.modelPath]=l.modelArtifactsInfo;e(c)},a.onerror=c=>(r.close(),t(a.error)),o.oncomplete=()=>r.close()},s.onerror=r=>t(s.error)})}async removeModel(e){return e=hh(e),new Promise((t,s)=>{const r=this.indexedDB.open(Fs,Ds);r.onupgradeneeded=()=>Os(r),r.onsuccess=()=>{const o=r.result,i=o.transaction(Je,"readwrite"),a=i.objectStore(Je),c=a.get(e);let l;c.onsuccess=()=>{if(c.result==null)return o.close(),s(new Error(`Cannot find model with path '${e}' in IndexedDB.`));{const u=a.delete(e),d=()=>{l=o.transaction(dt,"readwrite");const f=l.objectStore(dt).delete(e);f.onsuccess=()=>t(c.result.modelArtifactsInfo),f.onerror=p=>s(c.error)};u.onsuccess=d,u.onerror=h=>(d(),o.close(),s(c.error))}},c.onerror=u=>(o.close(),s(c.error)),i.oncomplete=()=>{l==null?o.close():l.oncomplete=()=>o.close()}},r.onerror=o=>s(r.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xe="/",Ot="tensorflowjs_models",Si="info",ph="model_topology",mh="weight_specs",gh="weight_data",xh="model_metadata";function Ii(n){return{info:[Ot,n,Si].join(Xe),topology:[Ot,n,ph].join(Xe),weightSpecs:[Ot,n,mh].join(Xe),weightData:[Ot,n,gh].join(Xe),modelMetadata:[Ot,n,xh].join(Xe)}}function Ri(n){for(const e of Object.values(n))window.localStorage.removeItem(e)}function Ch(n){const e=n.split(Xe);if(e.length<3)throw new Error(`Invalid key format: ${n}`);return e.slice(1,e.length-1).join(Xe)}function bh(n){return n.startsWith(xt.URL_SCHEME)?n.slice(xt.URL_SCHEME.length):n}class xt{constructor(e){if(!y().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,e==null||!e)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=Ii(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const t=JSON.stringify(e.modelTopology),s=JSON.stringify(e.weightSpecs),r=ts(e),o=St.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,ah(o));const i={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature!=null?e.signature:void 0,userDefinedMetadata:e.userDefinedMetadata!=null?e.userDefinedMetadata:void 0,modelInitializer:e.modelInitializer!=null?e.modelInitializer:void 0,initializerSignature:e.initializerSignature!=null?e.initializerSignature:void 0,trainingConfig:e.trainingConfig!=null?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(i)),{modelArtifactsInfo:r}}catch{throw Ri(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const t={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=s;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;const o=this.LS.getItem(this.keys.modelMetadata);if(o!=null){const a=JSON.parse(o);t.format=a.format,t.generatedBy=a.generatedBy,t.convertedBy=a.convertedBy,a.signature!=null&&(t.signature=a.signature),a.userDefinedMetadata!=null&&(t.userDefinedMetadata=a.userDefinedMetadata),a.modelInitializer!=null&&(t.modelInitializer=a.modelInitializer),a.initializerSignature!=null&&(t.initializerSignature=a.initializerSignature),a.trainingConfig!=null&&(t.trainingConfig=a.trainingConfig)}const i=this.LS.getItem(this.keys.weightData);if(i==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=ch(i),t}}xt.URL_SCHEME="localstorage://";const Ti=n=>y().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(xt.URL_SCHEME)?wh(n.slice(xt.URL_SCHEME.length)):null;Y.registerSaveRouter(Ti);Y.registerLoadRouter(Ti);function wh(n){return new xt(n)}class yh{constructor(){I(y().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),I(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const e={},t=Ot+Xe,s=Xe+Si;for(let r=0;r<this.LS.length;++r){const o=this.LS.key(r);if(o.startsWith(t)&&o.endsWith(s)){const i=Ch(o);e[i]=JSON.parse(this.LS.getItem(o))}}return e}async removeModel(e){e=bh(e);const t=Ii(e);if(this.LS.getItem(t.info)==null)throw new Error(`Cannot find model at path '${e}'`);const s=JSON.parse(this.LS.getItem(t.info));return Ri(t),s}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ur="://";class Ve{constructor(){this.managers={}}static getInstance(){return Ve.instance==null&&(Ve.instance=new Ve),Ve.instance}static registerManager(e,t){I(e!=null,()=>"scheme must not be undefined or null."),e.endsWith(Ur)&&(e=e.slice(0,e.indexOf(Ur))),I(e.length>0,()=>"scheme must not be an empty string.");const s=Ve.getInstance();I(s.managers[e]==null,()=>`A model store manager is already registered for scheme '${e}'.`),s.managers[e]=t}static getManager(e){const t=Ve.getInstance().managers[e];if(t==null)throw new Error(`Cannot find model manager for scheme '${e}'`);return t}static getSchemes(){return Object.keys(Ve.getInstance().managers)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $h{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>"u"||!y().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const r=this.functionRefs[s.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return ni(e)}}if(y().get("IS_BROWSER")){y().setPlatform("browser",new $h);try{Ve.registerManager(xt.URL_SCHEME,new yh)}catch{}try{Ve.registerManager(gt.URL_SCHEME,new fh)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vh={importFetch:()=>require("node-fetch")};let ps;class Sh{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return y().global.fetch!=null?y().global.fetch(e,t):(ps==null&&(ps=vh.importFetch()),ps(e,t))}now(){const e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?"":new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}}y().get("IS_NODE")&&!y().get("IS_BROWSER")&&y().setPlatform("node",new Sh);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ee(n,e="float32",t){return e=e||"float32",Cn(n),new Gn(n,e,t)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ih(n,e){const t=B(n,"x","cast");if(!Dc(e))throw new Error(`Failed to cast to unknown dtype ${e}`);if(e==="string"&&t.dtype!=="string"||e!=="string"&&t.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:t},r={dtype:e};return F.runKernel(Ys,s,r)}const Hn=H({cast_:Ih});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rh(n){const t={x:B(n,"x","clone","string_or_numeric")};return F.runKernel(Qs,t)}const Ei=H({clone_:Rh});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Th(n,e=!1){console.log(n.toString(e))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */xi();const Eh={buffer:ee,cast:Hn,clone:Ei,print:Th};jd(Eh);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nh(n,e){let t=B(n,"a","add"),s=B(e,"b","add");[t,s]=vt(t,s);const r={a:t,b:s};return F.runKernel(Ks,r)}const V=H({add_:Nh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kh(n,e){let t=B(n,"a","floorDiv"),s=B(e,"b","floorDiv");[t,s]=vt(t,s);const r={a:t,b:s};return F.runKernel(Mo,r)}const Ah=H({floorDiv_:kh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fh(n,e){let t=B(n,"a","div"),s=B(e,"b","div");if([t,s]=vt(t,s),t.dtype==="int32"&&s.dtype==="int32")return Ah(t,s);const r={a:t,b:s},o={};return F.runKernel(_o,r,o)}const We=H({div_:Fh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dh(n,e){let t=B(n,"a","mul"),s=B(e,"b","mul");[t,s]=vt(t,s);const r={a:t,b:s};return F.runKernel(Wo,r)}const P=H({mul_:Dh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oh(n){const e=B(n,"x","abs");if(e.dtype==="complex64"){const t={x:e};return F.runKernel(Po,t)}else{const t={x:e};return F.runKernel(Do,t)}}const Ph=H({abs_:Oh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ni(n,e,t,s,r="NHWC",o){const i=n[3],a=[...e,i],c=Kt(r);return Be(n,a,t,o,s,null,null,c)}function jt(n,e,t,s,r,o,i="channelsLast"){const[a,c]=dn(e);let l;if(i==="channelsLast")l=[a,c,n[3],n[3]];else if(i==="channelsFirst")l=[a,c,n[1],n[1]];else throw new Error(`Unknown dataFormat ${i}`);return Be(n,l,t,s,r,o,!1,i)}function bn(n,e,t,s,r,o,i="NDHWC"){const[a,c,l]=Ps(e);let u,d;if(i==="NDHWC")d="channelsLast",u=[a,c,l,n[4],n[4]];else if(i==="NCDHW")d="channelsFirst",u=[a,c,l,n[1],n[1]];else throw new Error(`Unknown dataFormat ${i}`);return wn(n,u,t,s,r,!1,d,o)}function Be(n,e,t,s,r,o,i=!1,a="channelsLast"){let[c,l,u,d]=[-1,-1,-1,-1];if(a==="channelsLast")[c,l,u,d]=n;else if(a==="channelsFirst")[c,d,l,u]=n;else throw new Error(`Unknown dataFormat ${a}`);const[h,f,,p]=e,[x,g]=dn(t),[m,C]=dn(s),b=Lt(h,m),w=Lt(f,C),{padInfo:$,outHeight:N,outWidth:T}=Bh(r,l,u,x,g,b,w,o,a),v=i?p*d:p;let D;return a==="channelsFirst"?D=[c,v,N,T]:a==="channelsLast"&&(D=[c,N,T,v]),{batchSize:c,dataFormat:a,inHeight:l,inWidth:u,inChannels:d,outHeight:N,outWidth:T,outChannels:v,padInfo:$,strideHeight:x,strideWidth:g,filterHeight:h,filterWidth:f,effectiveFilterHeight:b,effectiveFilterWidth:w,dilationHeight:m,dilationWidth:C,inShape:n,outShape:D,filterShape:e}}function wn(n,e,t,s,r,o=!1,i="channelsLast",a){let[c,l,u,d,h]=[-1,-1,-1,-1,-1];if(i==="channelsLast")[c,l,u,d,h]=n;else if(i==="channelsFirst")[c,h,l,u,d]=n;else throw new Error(`Unknown dataFormat ${i}`);const[f,p,x,,g]=e,[m,C,b]=Ps(t),[w,$,N]=Ps(s),T=Lt(f,w),v=Lt(p,$),D=Lt(x,N),{padInfo:O,outDepth:L,outHeight:M,outWidth:fe}=Mh(r,l,u,d,m,C,b,T,v,D,a),K=o?g*h:g;let ne;return i==="channelsFirst"?ne=[c,K,L,M,fe]:i==="channelsLast"&&(ne=[c,L,M,fe,K]),{batchSize:c,dataFormat:i,inDepth:l,inHeight:u,inWidth:d,inChannels:h,outDepth:L,outHeight:M,outWidth:fe,outChannels:K,padInfo:O,strideDepth:m,strideHeight:C,strideWidth:b,filterDepth:f,filterHeight:p,filterWidth:x,effectiveFilterDepth:T,effectiveFilterHeight:v,effectiveFilterWidth:D,dilationDepth:w,dilationHeight:$,dilationWidth:N,inShape:n,outShape:ne,filterShape:e}}function _h(n,e,t,s,r){s==null&&(s=tr(n,e,t));const o=n[0],i=n[1],a=hn((o-e+2*s)/t+1,r),c=hn((i-e+2*s)/t+1,r);return[a,c]}function Lh(n,e,t,s,r,o){r==null&&(r=tr(n,e[0],s[0]));const i=[0,0,0,t];for(let a=0;a<3;a++)n[a]+2*r>=e[a]&&(i[a]=hn((n[a]-e[a]+2*r)/s[a]+1,o));return i}function tr(n,e,t,s=1){const r=Lt(e,s);return Math.floor((n[0]*(t-1)-t+r)/2)}function dn(n){return typeof n=="number"?[n,n,n]:n.length===2?[n[0],n[1],1]:n}function Ps(n){return typeof n=="number"?[n,n,n]:n}function Lt(n,e){return e<=1?n:n+(n-1)*(e-1)}function Bh(n,e,t,s,r,o,i,a,c){let l,u,d;if(typeof n=="number"){l={top:n,bottom:n,left:n,right:n,type:n===0?"VALID":"NUMBER"};const f=_h([e,t],o,s,n,a);u=f[0],d=f[1]}else if(n==="same"){u=Math.ceil(e/s),d=Math.ceil(t/r);const h=Math.max(0,(u-1)*s+o-e),f=Math.max(0,(d-1)*r+i-t),p=Math.floor(h/2),x=h-p,g=Math.floor(f/2),m=f-g;l={top:p,bottom:x,left:g,right:m,type:"SAME"}}else if(n==="valid")l={top:0,bottom:0,left:0,right:0,type:"VALID"},u=Math.ceil((e-o+1)/s),d=Math.ceil((t-i+1)/r);else if(typeof n=="object"){const h=c==="channelsLast"?n[1][0]:n[2][0],f=c==="channelsLast"?n[1][1]:n[2][1],p=c==="channelsLast"?n[2][0]:n[3][0],x=c==="channelsLast"?n[2][1]:n[3][1];l={top:h,bottom:f,left:p,right:x,type:h===0&&f===0&&p===0&&x===0?"VALID":"EXPLICIT"},u=hn((e-o+h+f)/s+1,a),d=hn((t-i+p+x)/r+1,a)}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:l,outHeight:u,outWidth:d}}function Mh(n,e,t,s,r,o,i,a,c,l,u){let d,h,f,p;if(n==="valid"&&(n=0),typeof n=="number"){d={top:n,bottom:n,left:n,right:n,front:n,back:n,type:n===0?"VALID":"NUMBER"};const g=Lh([e,t,s,1],[a,c,l],1,[r,o,i],n,u);h=g[0],f=g[1],p=g[2]}else if(n==="same"){h=Math.ceil(e/r),f=Math.ceil(t/o),p=Math.ceil(s/i);const x=(h-1)*r+a-e,g=(f-1)*o+c-t,m=(p-1)*i+l-s,C=Math.floor(x/2),b=x-C,w=Math.floor(g/2),$=g-w,N=Math.floor(m/2),T=m-N;d={top:w,bottom:$,left:N,right:T,front:C,back:b,type:"SAME"}}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:d,outDepth:h,outHeight:f,outWidth:p}}function hn(n,e){if(!e)return Math.trunc(n);switch(e){case"round":return Math.round(n);case"ceil":return Math.ceil(n);case"floor":return Math.floor(n);default:throw new Error(`Unknown roundingMode ${e}`)}}function _s(n){const[e,t,s]=dn(n);return e===1&&t===1&&s===1}function qt(n,e){return _s(n)||_s(e)}function Vh(n){return dn(n).every(e=>e>0)}function Kt(n){if(n==="NHWC")return"channelsLast";if(n==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${n}`)}function Uh(n,e,t){if(t!=null){if(typeof e=="string")throw Error(`Error in ${n}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);if(typeof e=="number")I(Mn(e),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);else if(typeof e=="object")e.forEach(s=>{s.forEach(r=>{I(Mn(r),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${t} but got pad ${r}.`)})});else throw Error(`Error in ${n}: Unknown padding parameter: ${e}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wh(n,e){const s={x:B(n,"x","reshape","string_or_numeric")},r={shape:e};return F.runKernel(Xo,s,r)}const nr=H({reshape_:Wh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gh(n){const t={x:B(n,"x","sigmoid","float32")};return F.runKernel(qo,t)}const zh=H({sigmoid_:Gh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hh(n,e){let t=B(n,"broadcastTo","x");const s=t.shape;if(Cn(e),e.length<t.rank)throw new Error(`broadcastTo(): shape.length=${e.length} < input.rank=${t.rank}.`);if(e.length>t.rank){const l=t.shape.slice();for(;l.length<e.length;)l.unshift(1);t=nr(t,l)}const r=t.shape,o=Array.from(e);for(let l=e.length-1;l>=0;l--)if(r[l]===e[l])o[l]=1;else if(t.shape[l]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${e}].`);if(o.map((l,u)=>l>1?u:-1).filter(l=>l>=0).length===0)return Ei(t);const a={x:t},c={reps:o};return F.runKernel(Zo,a,c)}const Xh=H({broadcastTo_:Hh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jh(n,e,t){Cn(n),t=t||xn(e);const s={shape:n,value:e,dtype:t};return F.runKernel(Bo,{},s)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xn(n,e){const t=n.length,s=[];for(let r=0;r<t;r++){const o=t-1-r,i=n[o]||1;(e[e.length-1-r]||1)>1&&i===1&&s.unshift(o)}return s}function ki(n,e){const t=[];for(let s=0;s<e.length;s++){const r=n[n.length-s-1],o=e.length-s-1,i=e[o];(r==null||r===1&&i>1)&&t.unshift(o)}return t}function ae(n,e){const t=Math.max(n.length,e.length),s=new Array(t);for(let r=0;r<t;r++){let o=n[n.length-r-1];o==null&&(o=1);let i=e[e.length-r-1];if(i==null&&(i=1),o===1)s[t-r-1]=i;else if(i===1)s[t-r-1]=o;else if(o!==i){const a=`Operands could not be broadcast together with shapes ${n} and ${e}.`;throw Error(a)}else s[t-r-1]=o}return s}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qh(n){const t={x:B(n,"x","zerosLike")};return F.runKernel(Jo,t)}const Ge=H({zerosLike_:qh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kh(n){const t={x:B(n,"x","elu","float32")};return F.runKernel(Lo,t)}const Yh=H({elu_:Kh});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sr(n,e){for(let t=0;t<n.length;++t)if(n[n.length-t-1]!==e-1-t)return!1;return!0}function Ai(n,e,t){const s=n.length+e.length,r=[];let o=0,i=0;for(let a=0;a<s;a++)t.indexOf(a)===-1?r.push(n[o++]):r.push(e[i++]);return r}function He(n,e){const t=[],s=n.length;for(let o=0;o<s;o++)e.indexOf(o)===-1&&t.push(n[o]);const r=e.map(o=>n[o]);return[t,r]}function je(n,e){const t=e.map(s=>1);return Ai(n,t,e)}function Me(n,e,t){I(sr(e,t),()=>`${n} supports only inner-most axes for now. Got axes ${e} and rank-${t} input.`)}function Te(n,e){if(sr(n,e))return null;const t=[];for(let s=0;s<e;++s)n.indexOf(s)===-1&&t.push(s);return n.forEach(s=>t.push(s)),t}function rr(n){return n.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function Ee(n,e){const t=[];for(let s=e-n;s<e;++s)t.push(s);return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qh(n,e){let t=B(n,"base","pow"),s=B(e,"exp","pow");[t,s]=vt(t,s);const r={a:t,b:s};return F.runKernel(Go,r)}const Wr=H({pow_:Qh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function st(n,e){if((Re(n)&&e!=="string"||Array.isArray(n))&&e!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(e==="string"&&Re(n)&&!(n instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return sh(n,[],[],e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zh(n){const t={x:B(n,"x","sqrt","float32")};return F.runKernel(Ko,t)}const Wt=H({sqrt_:Zh});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jh(n){const e=B(n,"x","square"),t={};return F.runKernel("Square",{x:e},t)}const ft=H({square_:Jh});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ef(n,e=null,t=!1){let s=B(n,"x","sum");s.dtype==="bool"&&(s=Hn(s,"int32"));const r={x:s},o={axis:e,keepDims:t};return F.runKernel(Yo,r,o)}const tf=H({sum_:ef});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nf(n,e=.2){const s={x:B(n,"x","leakyRelu")},r={alpha:e};return F.runKernel(Vo,s,r)}const sf=H({leakyRelu_:nf});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rf(n,e){I(vs(n),()=>"The f passed in variableGrads(f) must be a function"),I(e==null||Array.isArray(e)&&e.every(l=>l instanceof zn),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const t=e!=null;if(!t){e=[];for(const l in F.registeredVariables)e.push(F.registeredVariables[l])}const s=t?e.filter(l=>!l.trainable):null,r=e.length;e=e.filter(l=>l.trainable),I(e.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const o=!0,{value:i,grads:a}=F.gradients(n,e,null,o);I(a.some(l=>l!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),I(i.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${i.rank} tensor`);const c={};return e.forEach((l,u)=>{a[u]!=null&&(c[l.name]=a[u])}),s!=null&&s.forEach(l=>c[l.name]=null),{value:i,grads:c}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function of(n,e){let t=B(n,"a","sub"),s=B(e,"b","sub");[t,s]=vt(t,s);const r={a:t,b:s};return F.runKernel(Qo,r)}const Bt=H({sub_:of});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function af(n,e){let t=B(n,"a","maximum"),s=B(e,"b","maximum");[t,s]=vt(t,s),t.dtype==="bool"&&(t=Hn(t,"int32"),s=Hn(s,"int32")),ae(t.shape,s.shape);const r={a:t,b:s};return F.runKernel(Uo,r)}const cf=H({maximum_:af});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ls(n,e="float32"){if(Cn(n),e==="complex64"){const s=Ls(n,"float32"),r=Ls(n,"float32");return nh(s,r)}const t=nt(E(n),e);return F.makeTensor(t,n,e)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lf(n,e){const t=B(n,"x","prelu"),s=B(e,"alpha","prelu"),r={x:t,alpha:s};return F.runKernel(zo,r)}const uf=H({prelu_:lf});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function df(n){const t={x:B(n,"x","relu")};return F.runKernel(Ho,t)}const hf=H({relu_:df});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ff(n){const t={x:B(n,"x","relu6")};return F.runKernel(jo,t)}const pf=H({relu6_:ff});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mf(n,e=0){const s={x:B(n,"x","step")},r={alpha:e};return F.runKernel(ei,s,r)}const gf=H({step_:mf});function Fi(n,e,t){const s=e.rank>1?e.shape[e.rank-1]:1,r=e.rank>1?e.rank-1:1,o=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${t.shape}, indices.shape: ${e.shape}, shape: ${n}, sliceDim: ${s}, and batchDim: ${r}.`;if(t.rank<r)throw new Error(o+` update.rank < ${r}. `);if(n.length<s+(t.rank-r))throw new Error(o+` Output shape length < ${s+(t.rank-r)}`);if(t.rank!==r+n.length-s)throw new Error(o+` update.rank != ${r+n.length-s}`);for(let i=0;i<r;++i)if(t.shape[i]!==e.shape[i])throw new Error(o+` updates.shape[${i}] (${t.shape[i]}) != indices.shape[${i}] (${e.shape[i]}).`);for(let i=0;i<t.rank-r;++i)if(t.shape[i+r]!==n[i+s])throw new Error(o+` updates.shape[${i+r}] (${t.shape[i+r]}) != shape[${i+r}] (${n[i+r]})`)}function xf(n,e,t){if(e.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${e.rank}.`);if(n.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${n.rank}.`);if(e.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${e.dtype}`);if(t.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${t}`);if(t.length===0){if(e.size===0)throw new Error(`Indices specified for empty output. indices shape: ${e.shape}`);if(n.size===0)throw new Error(`Updates specified for empty output. updates shape: ${n.shape}`)}Fi(t,e,n)}function ns(n,e,t){const s=e.shape.length,r=s>1?e.shape[s-1]:1,o=t.length;let i=1;for(let d=r;d<o;++d)i*=t[d];const a=r<1?1:r,c=E(e.shape)/a,l=[...Z(t.slice(0,r)),1],u=E(t);return{sliceRank:r,numUpdates:c,sliceSize:i,strides:l,outputSize:u}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cf(n,e){const t=[];for(let o=0;o<e.length;o++)e[o]&&t.push(o);const s=ee(n,"int32"),r=ee([t.length,n.length],"int32");for(let o=0;o<t.length;o++){const i=s.indexToLoc(t[o]),a=o*n.length;r.values.set(i,a)}return r.toTensor()}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bf(n,e,t){if(t==null||t==="linear")return n;if(t==="relu")return P(n,gf(e));throw new Error(`Cannot compute gradient for fused activation ${t}.`)}function wf(n,e){let t=e;const s=ki(n.shape,e.shape);return s.length>0&&(t=tf(t,s)),nr(t,n.shape)}function yf(n,e,t,s){if(e==="linear")return n;if(e==="relu")return hf(n);if(e==="elu")return Yh(n);if(e==="relu6")return pf(n);if(e==="prelu")return uf(n,t);if(e==="leakyrelu")return sf(n,s);if(e==="sigmoid")return zh(n);throw new Error(`Unknown fused activation ${e}.`)}const $f=(n,e)=>!(n>0)||e==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vf(n,e,t){const s=Sf(n,e,t),r=s<0?-(s+1):s;n.splice(r,0,e)}function Sf(n,e,t){return Rf(n,e,t||If)}function If(n,e){return n>e?1:n<e?-1:0}function Rf(n,e,t){let s=0,r=n.length,o=0,i=!1;for(;s<r;){o=s+(r-s>>>1);const a=t(e,n[o]);a>0?s=o+1:(r=o,i=!a)}return i?s:-s-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tf(n,e,t,s,r){return or(n,e,t,s,r,0)}function Ef(n,e,t,s,r,o){return or(n,e,t,s,r,0,!1,o,!0)}function Nf(n,e,t,s,r,o){return or(n,e,t,s,r,o,!0)}function or(n,e,t,s,r,o,i=!1,a=!1,c=!1){const l=[];for(let g=0;g<e.length;g++)e[g]>r&&l.push({score:e[g],boxIndex:g,suppressBeginIndex:0});l.sort(Gr);const u=o>0?-.5/o:0,d=[],h=[];for(;d.length<t&&l.length>0;){const g=l.pop(),{score:m,boxIndex:C,suppressBeginIndex:b}=g;if(m<r)break;let w=!1;for(let $=d.length-1;$>=b;--$){const N=kf(n,C,d[$]);if(N>=s){w=!0;break}if(g.score=g.score*Af(s,u,N),g.score<=r)break}g.suppressBeginIndex=d.length,w||(g.score===m?(d.push(C),h.push(g.score)):g.score>r&&vf(l,g,Gr))}const f=d.length,p=t-f;a&&p>0&&(d.push(...new Array(p).fill(0)),h.push(...new Array(p).fill(0)));const x={selectedIndices:d};return i&&(x.selectedScores=h),c&&(x.validOutputs=f),x}function kf(n,e,t){const s=n.subarray(e*4,e*4+4),r=n.subarray(t*4,t*4+4),o=Math.min(s[0],s[2]),i=Math.min(s[1],s[3]),a=Math.max(s[0],s[2]),c=Math.max(s[1],s[3]),l=Math.min(r[0],r[2]),u=Math.min(r[1],r[3]),d=Math.max(r[0],r[2]),h=Math.max(r[1],r[3]),f=(a-o)*(c-i),p=(d-l)*(h-u);if(f<=0||p<=0)return 0;const x=Math.max(o,l),g=Math.max(i,u),m=Math.min(a,d),C=Math.min(c,h),b=Math.max(m-x,0)*Math.max(C-g,0);return b/(f+p-b)}function Af(n,e,t){const s=Math.exp(e*t*t);return t<=n?s:0}function Gr(n,e){return n.score-e.score||n.score===e.score&&e.boxIndex-n.boxIndex}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ff=new Map,Df=new Map;class Of{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}}class ct{constructor(){this.classNameMap={}}static getMap(){return ct.instance==null&&(ct.instance=new ct),ct.instance}static register(e){ct.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function Pf(n,e,t){I(n.className!=null,()=>"Class being registered does not have the static className property defined."),I(typeof n.className=="string",()=>"className is required to be a string, but got type "+typeof n.className),I(n.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof e>"u"&&(e="Custom"),typeof t>"u"&&(t=n.className);const s=t,r=e+">"+s;return ct.register(n),Ff.set(r,n),Df.set(n,r),n}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class It extends Of{minimize(e,t=!1,s){const{value:r,grads:o}=this.computeGradients(e,s);if(s!=null){const i=s.map(a=>({name:a.name,tensor:o[a.name]}));this.applyGradients(i)}else this.applyGradients(o);return be(o),t?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return rf(e,t)}dispose(){this.iterations_!=null&&be(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:st(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(It,Symbol.hasInstance,{value:n=>n.minimize!=null&&n.computeGradients!=null&&n.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class _f extends It{static get className(){return"Adadelta"}constructor(e,t,s=null){super(),this.learningRate=e,this.rho=t,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=F.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const o=F.registeredVariables[s],i=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accum_grad`,variable:X(()=>Ge(o).variable(i))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${s}/accum_var`,variable:X(()=>Ge(o).variable(i))});const a=Array.isArray(e)?e[r].tensor:e[s];if(a==null)return;const c=this.accumulatedGrads[r].variable,l=this.accumulatedUpdates[r].variable;X(()=>{const u=V(P(c,this.rho),P(ft(a),1-this.rho)),d=P(We(Wt(V(l,this.epsilon)),Wt(V(c,this.epsilon))),a),h=V(P(l,this.rho),P(ft(d),1-this.rho));c.assign(u),l.assign(h);const f=V(P(d,-this.learningRate),o);o.assign(f)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(be(this.accumulatedGrads.map(e=>e.variable)),be(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){const e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=e.length/2,s=!1;this.accumulatedGrads=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedUpdates=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Lf extends It{static get className(){return"Adagrad"}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const o=F.registeredVariables[s];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accumulator`,variable:X(()=>jh(o.shape,this.initialAccumulatorValue).variable(!1))});const i=Array.isArray(e)?e[r].tensor:e[s];if(i==null)return;const a=this.accumulatedGrads[r].variable;X(()=>{const c=V(a,ft(i));a.assign(c);const l=V(P(We(i,Wt(V(c,F.backend.epsilon()))),-this.learningRate),o);o.assign(l)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&be(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulatedGrads=e.map(s=>({originalName:s.name,variable:s.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bf extends It{static get className(){return"Adam"}constructor(e,t,s,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=s,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],X(()=>{this.accBeta1=st(t).variable(),this.accBeta2=st(s).variable()}),r==null&&(this.epsilon=F.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(s=>s.name):Object.keys(e);X(()=>{const s=Bt(1,this.accBeta1),r=Bt(1,this.accBeta2);t.forEach((o,i)=>{const a=F.registeredVariables[o],c=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:X(()=>Ge(a).variable(c))}),this.accumulatedSecondMoment[i]==null&&(this.accumulatedSecondMoment[i]={originalName:`${o}/v`,variable:X(()=>Ge(a).variable(c))});const l=Array.isArray(e)?e[i].tensor:e[o];if(l==null)return;const u=this.accumulatedFirstMoment[i].variable,d=this.accumulatedSecondMoment[i].variable,h=V(P(u,this.beta1),P(l,1-this.beta1)),f=V(P(d,this.beta2),P(ft(l),1-this.beta2)),p=We(h,s),x=We(f,r);u.assign(h),d.assign(f);const g=V(P(We(p,V(Wt(x),this.epsilon)),-this.learningRate),a);a.assign(g)}),this.accBeta1.assign(P(this.accBeta1,this.beta1)),this.accBeta2.assign(P(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&be(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&be(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e),X(()=>{this.accBeta1.assign(Wr(this.beta1,this.iterations_+1)),this.accBeta2.assign(Wr(this.beta2,this.iterations_+1))});const t=e.length/2,s=!1;this.accumulatedFirstMoment=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Mf extends It{static get className(){return"Adamax"}constructor(e,t,s,r=null,o=0){super(),this.learningRate=e,this.beta1=t,this.beta2=s,this.epsilon=r,this.decay=o,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],X(()=>{this.iteration=st(0).variable(),this.accBeta1=st(t).variable()}),r==null&&(this.epsilon=F.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(s=>s.name):Object.keys(e);X(()=>{const s=Bt(1,this.accBeta1),r=We(-this.learningRate,V(P(this.iteration,this.decay),1));t.forEach((o,i)=>{const a=F.registeredVariables[o],c=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:Ge(a).variable(c)}),this.accumulatedWeightedInfNorm[i]==null&&(this.accumulatedWeightedInfNorm[i]={originalName:`${o}/v`,variable:Ge(a).variable(c)});const l=Array.isArray(e)?e[i].tensor:e[o];if(l==null)return;const u=this.accumulatedFirstMoment[i].variable,d=this.accumulatedWeightedInfNorm[i].variable,h=V(P(u,this.beta1),P(l,1-this.beta1)),f=P(d,this.beta2),p=Ph(l),x=cf(f,p);u.assign(h),d.assign(x);const g=V(P(We(r,s),We(h,V(x,this.epsilon))),a);a.assign(g)}),this.iteration.assign(V(this.iteration,1)),this.accBeta1.assign(P(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&be(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&be(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Di extends It{static get className(){return"SGD"}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const o=Array.isArray(e)?e[r].tensor:e[s];if(o==null)return;const i=F.registeredVariables[s];X(()=>{const a=V(P(this.c,o),i);i.assign(a)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=oh(st(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Vf extends Di{static get className(){return"Momentum"}constructor(e,t,s=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=s,this.accumulations=[],this.m=st(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const o=F.registeredVariables[s];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${s}/momentum`,variable:X(()=>Ge(o).variable(!1))});const i=this.accumulations[r].variable,a=Array.isArray(e)?e[r].tensor:e[s];a!=null&&X(()=>{let c;const l=V(P(this.m,i),a);this.useNesterov?c=V(P(this.c,V(a,P(l,this.m))),o):c=V(P(this.c,l),o),i.assign(l),o.assign(c)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&be(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulations=e.map(s=>({originalName:s.name,variable:s.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Uf extends It{static get className(){return"RMSProp"}constructor(e,t=.9,s=0,r=null,o=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=s,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=o,r==null&&(this.epsilon=F.backend.epsilon()),e==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const o=F.registeredVariables[s],i=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${s}/rms`,variable:X(()=>Ge(o).variable(i))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${s}/momentum`,variable:X(()=>Ge(o).variable(i))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${s}/mg`,variable:X(()=>Ge(o).variable(i))});const a=Array.isArray(e)?e[r].tensor:e[s];if(a==null)return;const c=this.accumulatedMeanSquares[r].variable,l=this.accumulatedMoments[r].variable;X(()=>{const u=V(P(c,this.decay),P(ft(a),1-this.decay));if(this.centered){const d=this.accumulatedMeanGrads[r].variable,h=V(P(d,this.decay),P(a,1-this.decay)),f=We(P(a,this.learningRate),Wt(Bt(u,V(ft(h),this.epsilon)))),p=V(P(l,this.momentum),f);c.assign(u),d.assign(h),l.assign(p);const x=Bt(o,p);o.assign(x)}else{const d=V(P(c,this.decay),P(ft(a),1-this.decay)),h=V(P(l,this.momentum),We(P(a,this.learningRate),Wt(V(d,this.epsilon))));c.assign(d),l.assign(h);const f=Bt(o,h);o.assign(f)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&be(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&be(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&be(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=this.centered?e.length/3:e.length/2,s=!1;this.accumulatedMeanSquares=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedMoments=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wf=[_f,Lf,Bf,Mf,Vf,Uf,Di];function Gf(){for(const n of Wf)Pf(n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zf="model",Hf=".json",Xf=".weights.bin";function zr(n){return new Promise(e=>setTimeout(e)).then(n)}class Ct{constructor(e){if(!y().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(Ct.URL_SCHEME)&&(e=e.slice(Ct.URL_SCHEME.length)),(e==null||e.length===0)&&(e=zf),this.modelJsonFileName=e+Hf,this.weightDataFileName=e+Xf}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=St.join(e.weightData),s=window.URL.createObjectURL(new Blob([t],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],o=yi(e,r),i=window.URL.createObjectURL(new Blob([JSON.stringify(o)],{type:"application/json"})),a=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(a.download=this.modelJsonFileName,a.href=i,await zr(()=>a.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const c=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;c.download=this.weightDataFileName,c.href=s,await zr(()=>c.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:ts(e)}}}}Ct.URL_SCHEME="downloads://";const jf=n=>y().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(Ct.URL_SCHEME)?qf(n.slice(Ct.URL_SCHEME.length)):null;Y.registerSaveRouter(jf);function qf(n="model"){return new Ct(n)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hr(n,e,t,s){i(n),t=t??0,s=s??1,a(t,s);let r=0;const o=c=>(c.then(l=>{const u=t+ ++r/n.length*(s-t);return e(u),l}),c);function i(c){I(c!=null&&Array.isArray(c)&&c.length>0,()=>"promises must be a none empty array")}function a(c,l){I(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${c}`),I(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${l}`),I(l>=c,()=>`startFraction must be no more than endFraction, but got startFraction ${c} and endFraction ${l}`)}return Promise.all(n.map(o))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Kf(n,e){e==null&&(e={});const t=e.fetchFunc==null?y().platform.fetch:e.fetchFunc,s=n.map(d=>t(d,e.requestInit,{isBinary:!0})),a=(e.onProgress==null?await Promise.all(s):await Hr(s,e.onProgress,0,.5)).map(d=>d.arrayBuffer());return e.onProgress==null?await Promise.all(a):await Hr(a,e.onProgress,.5,1)}function Yf(n,e){var t;const s=e.fetchFunc==null?y().platform.fetch:e.fetchFunc;let r=0,o;return(t=e.onProgress)===null||t===void 0||t.call(e,0),new ReadableStream({pull:async i=>{for(var a;r<n.length;){o||(o=(await s(n[r],e.requestInit,{isBinary:!0})).body.getReader());const{done:c,value:l}=await o.read();if(c){r++,o=void 0,(a=e.onProgress)===null||a===void 0||a.call(e,r/n.length);continue}i.enqueue(l);return}i.close()}})}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qf="application/octet-stream",Zf="application/json";class ir{constructor(e,t){if(this.DEFAULT_METHOD="POST",t==null&&(t={}),this.weightPathPrefix=t.weightPathPrefix,this.weightUrlConverter=t.weightUrlConverter,t.fetchFunc!=null?(I(typeof t.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc):this.fetch=y().platform.fetch,I(e!=null&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&I(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,t.requestInit!=null&&t.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=t.requestInit||{},this.loadOptions=t}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:e.weightSpecs}],r=yi(e,s);if(t.body.append("model.json",new Blob([JSON.stringify(r)],{type:Zf}),"model.json"),e.weightData!=null){const i=St.join(e.weightData);t.body.append("model.weights.bin",new Blob([i],{type:Qf}),"model.weights.bin")}const o=await this.fetch(this.path,t);if(o.ok)return{modelArtifactsInfo:ts(e),responses:[o]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${o.status}.`)}async loadModelJSON(){const e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw new Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch{let i=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?i+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":i+=" Please make sure the server is serving valid JSON for this request.",new Error(i)}const s=t.modelTopology,r=t.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return t}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const e=await this.loadModelJSON();return uh(e,t=>this.loadWeights(t))}async loadStream(){const e=await this.loadModelJSON(),t=await this.getWeightUrls(e.weightsManifest),s=Vr(e.weightsManifest),r=()=>Yf(t,this.loadOptions);return Object.assign(Object.assign({},e),{weightSpecs:s,getWeightStream:r})}async getWeightUrls(e){const t=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=Jf(t),o=this.weightPathPrefix||s,i=[],a=[];for(const c of e)for(const l of c.paths)this.weightUrlConverter!=null?a.push(this.weightUrlConverter(l)):i.push(o+l+r);return this.weightUrlConverter&&i.push(...await Promise.all(a)),i}async loadWeights(e){const t=await this.getWeightUrls(e),s=Vr(e),r=await Kf(t,this.loadOptions);return[s,r]}}ir.URL_SCHEME_REGEX=/^https?:\/\//;function Jf(n){const e=n.lastIndexOf("/"),t=n.lastIndexOf("?"),s=n.substring(0,e),r=t>e?n.substring(t):"";return[s+"/",r]}function Xr(n){return n.match(ir.URL_SCHEME_REGEX)!=null}const Oi=(n,e)=>{if(typeof fetch>"u"&&(e==null||e.fetchFunc==null))return null;{let t=!0;if(Array.isArray(n)?t=n.every(s=>Xr(s)):t=Xr(n),t)return ep(n,e)}return null};Y.registerSaveRouter(Oi);Y.registerLoadRouter(Oi);function ep(n,e){return new ir(n,e)}function Pi(n,e){const t=n.shape.length,s=e.shape.length;if(t<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${t}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(e.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.shape[s-1]>t)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${e.shape[s-1]} vs. ${t}`);if(E(n.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${n.shape}.`);const r=e.shape,o=r[r.length-1];let i=1;for(let d=0;d<r.length-1;++d)i*=r[d];const a=n.shape,c=r.slice();c.pop();let l=1;for(let d=o;d<t;++d)l*=a[d],c.push(a[d]);const u=[...Z(n.shape).map(d=>d/l),1].slice(0,o);return[c,i,l,u]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bs=-2,tp=-1;function _i(n,e,t){const s=n.shape.length;I(s===e.length,()=>`Error in slice${s}D: Length of begin ${e} must match the rank of the array (${s}).`),I(s===t.length,()=>`Error in slice${s}D: Length of size ${t} must match the rank of the array (${s}).`);for(let r=0;r<s;++r)I(e[r]+t[r]<=n.shape[r],()=>`Error in slice${s}D: begin[${r}] + size[${r}] (${e[r]+t[r]}) would overflow input.shape[${r}] (${n.shape[r]})`)}function np(n){const e=[];let t=0;for(;n>0;)n&1&&e.push(t),n/=2,t++;return e}function Li(n,e,t){const s=[];for(let r=0;r<n.length;r++)s[r]=Math.ceil((e[r]-n[r])/t[r]);return s}function Bi(n,e,t,s){const r=[...n];for(let o=r.length;o<s.length;o++)r.push(1);for(let o=0;o<t;o++)o===0?r[e]=1:(r.splice(e,0,1),r.pop());return r}function Mi(n,e,t){return t<=n?t:t-(e-1)}function Vi(n,e){const t=[];for(let s=0;s<n;s++)t.push(e+s);return t}function sp(n,e,t,s,r,o,i,a,c){const l=n.length;let u=new Array(l),d=new Array(l),h=new Array(l);if(e.length&&t>0){const f=e[0],p=t+1;u=Ui(i,f,p,s,n),d=Wi(a,f,p,r,n),h=Bi(o,f,p,n)}else for(let f=0;f<l;f++)u[f]=zi(i,s,o,n,f,c),d[f]=Hi(a,r,o,n,f,c),h[f]=Gi(o,f,c);return{begin:u,end:d,strides:h}}function Ui(n,e,t,s,r){const o=[...r],i=Vi(t,e);for(let a=0;a<o.length;a++)if(i.indexOf(a)>-1)o[a]=0;else{const c=Mi(e,t,a);let l=s[c];n&1<<c&&(l=0),o[a]=l}return o}function Wi(n,e,t,s,r){const o=[...r],i=Vi(t,e);for(let a=0;a<o.length;a++)if(i.indexOf(a)>-1)o[a]=Number.MAX_SAFE_INTEGER;else{const c=Mi(e,t,a);let l=s[c];n&1<<c&&(l=Number.MAX_SAFE_INTEGER),o[a]=l}for(let a=0;a<o.length;a++){const c=r[a];o[a]<0&&(o[a]+=c),o[a]=Bn(0,o[a],r[a])}return o}function Gi(n,e,t){let s=n[e];return(t&1<<e||s==null)&&(s=1),s}function zi(n,e,t,s,r,o){let i=e[r];const a=t[r]||1;(n&1<<r||o&1<<r||i==null)&&(a>0?i=Number.MIN_SAFE_INTEGER:i=Number.MAX_SAFE_INTEGER);const c=s[r];return i<0&&(i+=c),i=Bn(0,i,c-1),i}function Hi(n,e,t,s,r,o){let i=e[r];const a=t[r]||1;(n&1<<r||o&1<<r||i==null)&&(a>0?i=Number.MAX_SAFE_INTEGER:i=Number.MIN_SAFE_INTEGER);const c=s[r];return i<0&&(i+=c),a>0?i=Bn(0,i,c):i=Bn(-1,i,c-1),i}function ar(n,e,t){let s=t.length;for(let r=0;r<t.length;r++)if(t[r]>1){s=r;break}for(let r=s+1;r<t.length;r++)if(e[r]>0||t[r]!==n[r])return!1;return!0}function cr(n,e){let t=n.length>0?n[n.length-1]:1;for(let s=0;s<n.length-1;s++)t+=n[s]*e[s];return t}function Xi(n,e,t){let s;const r=n.shape.length;typeof e=="number"?s=[e,...new Array(r-1).fill(0)]:e.length<r?s=e.concat(new Array(r-e.length).fill(0)):s=e.slice(),s.forEach(i=>{I(i!==-1,()=>"slice() does not support negative begin indexing.")});let o;return t==null?o=new Array(r).fill(-1):typeof t=="number"?o=[t,...new Array(r-1).fill(-1)]:t.length<r?o=t.concat(new Array(r-t.length).fill(-1)):o=t,o=o.map((i,a)=>i>=0?i:(I(i===-1,()=>`Negative size values should be exactly -1 but got ${i} for the slice() size at index ${a}.`),n.shape[a]-s[a])),[s,o]}function ji(n,e,t,s,r,o,i,a,c){let l;if(s==null?(l=new Array(e.length),l.fill(1)):l=s,i!=null&&i&i-1)throw new Error("Multiple ellipses in slice is not allowed.");let u=!1;const d={dims:l.length,numAddAxisAfterEllipsis:0,begin:e.slice(),end:t.slice(),strides:l.slice(),beginMask:r,endMask:o,ellipsisMask:i,newAxisMask:a,shrinkAxisMask:c};for(let b=0;b<d.dims;b++)u&&1<<b&a&&d.numAddAxisAfterEllipsis++,1<<b&i&&(u=!0);u||(d.ellipsisMask|=1<<d.dims,d.dims++);const h={dims:n.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};rp(d,h);let f=!0,p=!0,x=!0;const g=[],m=[];for(let b=0;b<n.length;++b){if(h.strides[b]===0)throw Error(`strides[${b}] must be non-zero`);const w=!!(h.shrinkAxisMask&1<<b),$=n[b];if($===-1){g.push(w?1:-1);continue}const N=[h.beginMask&1<<b,h.endMask&1<<b],T=[h.strides[b]>0?0:-1,h.strides[b]>0?$:$-1];if(w&&h.strides[b]<=0)throw Error("only stride 1 allowed on non-range indexing.");x=x&&h.strides[b]===1;const v=!!(h.beginMask&1<<b&&h.endMask&1<<b);if(h.beginValid&&h.endValid){if(w){const M=h.begin[b]<0?$+h.begin[b]:h.begin[b];if(h.begin[b]=M,h.end[b]=h.begin[b]+1,M<0||M>=$)throw Error(`slice index ${h.begin[b]} of dimension ${b} out of bounds.`)}else h.begin[b]=jr(h.begin[b],0,h.strides[b],$,N,T),h.end[b]=jr(h.end[b],1,h.strides[b],$,N,T);const L=h.strides[b]===1&&h.begin[b]===0&&h.end[b]===$;f=f&&L,p=p&&(b===0&&h.strides[b]===1||L)}else f=f&&h.strides[b]===1&&v,p=p&&(b===0&&h.strides[b]===1||v);let D,O=!1;if(h.beginValid&&h.endValid?(D=h.end[b]-h.begin[b],O=!0):w?(D=1,O=!0):v&&$>=0&&(h.strides[b]<0?D=-$:D=$,O=!0),O){let L;D===0||D<0!=h.strides[b]<0?L=0:L=Math.trunc(D/h.strides[b])+(D%h.strides[b]!==0?1:0),g.push(L)}else g.push(-1)}for(let b=0;b<h.finalShapeGatherIndices.length;++b){const w=h.finalShapeGatherIndices[b];w>=0?m.push(g[w]):w===Bs&&m.push(1)}return{finalShapeSparse:m.filter((b,w)=>h.finalShapeGatherIndices[w]!==Bs),finalShape:m,isIdentity:f,sliceDim0:p,isSimpleSlice:x,begin:h.begin,end:h.end,strides:h.strides}}function rp(n,e){e.beginMask=0,e.endMask=0,e.shrinkAxisMask=0;let t=0;e.beginValid=n.begin!=null,e.endValid=n.end!=null,e.begin=new Array(e.dims),e.end=new Array(e.dims),e.strides=new Array(e.dims),e.finalShapeGatherIndices=[],e.finalShapeGatherIndicesSparse=[],e.inputShapeGatherIndicesSparse=new Array(e.dims);for(let s=0;s<n.dims;s++)if(1<<s&n.ellipsisMask){const r=Math.min(e.dims-(n.dims-s)+1+n.numAddAxisAfterEllipsis,e.dims);for(;t<r;t++)e.begin[t]=0,e.end[t]=0,e.strides[t]=1,e.beginMask|=1<<t,e.endMask|=1<<t,e.finalShapeGatherIndices.push(t),e.finalShapeGatherIndicesSparse.push(-1),e.inputShapeGatherIndicesSparse[t]=s}else if(1<<s&n.newAxisMask)e.finalShapeGatherIndices.push(Bs),e.finalShapeGatherIndicesSparse.push(-1);else{if(t===e.begin.length)throw Error(`Index out of range using input dim ${t}; input has only ${e.dims} dims, ${e.begin.length}.`);n.begin!=null&&(e.begin[t]=n.begin[s]),n.end!=null&&(e.end[t]=n.end[s]),e.strides[t]=n.strides[s],n.beginMask&1<<s&&(e.beginMask|=1<<t),n.endMask&1<<s&&(e.endMask|=1<<t),n.shrinkAxisMask&1<<s?(e.finalShapeGatherIndices.push(tp),e.finalShapeGatherIndicesSparse.push(-1),e.shrinkAxisMask|=1<<t):(e.finalShapeGatherIndices.push(t),e.finalShapeGatherIndicesSparse.push(s)),e.inputShapeGatherIndicesSparse[t]=s,t++}}function jr(n,e,t,s,r,o){if(r[e])return t>0?o[e]:o[e+1&1];{const i=n<0?s+n:n;return i<o[0]?o[0]:i>o[1]?o[1]:i}}const op=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:_i,computeFlatOffset:cr,computeOutShape:Li,getNormalizedAxes:sp,isSliceContinous:ar,maskToAxes:np,parseSliceParams:Xi,sliceInfo:ji,startForAxis:zi,startIndicesWithElidedDims:Ui,stopForAxis:Hi,stopIndicesWithElidedDims:Wi,stridesForAxis:Gi,stridesWithElidedDims:Bi},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ip=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:n=>n();function ap(){return new Promise(n=>ip(()=>n()))}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qi(n,e){const t=n[0].length;n.forEach((r,o)=>{I(r.length===t,()=>`Error in concat${t}D: rank of tensors[${o}] must be the same as the rank of the rest (${t})`)}),I(e>=0&&e<t,()=>`Error in concat${t}D: axis must be between 0 and ${t-1}.`);const s=n[0];n.forEach((r,o)=>{for(let i=0;i<t;i++)I(i===e||r[i]===s[i],()=>`Error in concat${t}D: Shape of tensors[${o}] (${r}) does not match the shape of the rest (${s}) along the non-concatenated axis ${o}.`)})}function bt(n,e){const t=n[0].slice();for(let s=1;s<n.length;s++)t[e]+=n[s][e];return t}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Oe;(function(n){n[n.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",n[n.VALUE_ROWIDS=1]="VALUE_ROWIDS",n[n.ROW_LENGTHS=2]="ROW_LENGTHS",n[n.ROW_SPLITS=3]="ROW_SPLITS",n[n.ROW_LIMITS=4]="ROW_LIMITS",n[n.ROW_STARTS=5]="ROW_STARTS"})(Oe||(Oe={}));function Ki(n,e,t){let s=new Array;if(t==null&&e==null)return s;if(e==null)for(;s.length<n+t.length;)s.push(-1);else s=e.slice();if(t==null)return s;if(n+t.length!==s.length)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.rank = ${n+t.length}, but shape.rank = ${s.length}`);for(let r=1;r<t.length;++r){const o=t[r],i=s[s.length-t.length+r],a=s[i];if(o>=0)if(a>=0){if(a!==o)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.shape[${r+n}] = ${o} but shape[${r+n}] = ${a}`)}else s[i]=o}return s}function Yi(n){const e={FIRST_DIM_SIZE:Oe.FIRST_DIM_SIZE,VALUE_ROWIDS:Oe.VALUE_ROWIDS,ROW_LENGTHS:Oe.ROW_LENGTHS,ROW_SPLITS:Oe.ROW_SPLITS,ROW_LIMITS:Oe.ROW_LIMITS,ROW_STARTS:Oe.ROW_STARTS},t=[];for(const s of n)if(s in e)t.push(e[s]);else break;return t}function Qi(n){return n.length===0?0:n[0]===Oe.FIRST_DIM_SIZE?n.length-1:n.length}function Zi(n,e){if(n==null||e==null)return;const t=n.length,s=e.length;if(t>=s)throw new Error(`defaultValue.shape=${n} and ragged tensor flatValues.shape=${e}, are incompatible: defaultValue.rank = ${t} must be less than ragged tensor input flatValues.rank = ${s})`);for(let r=0;r<Math.min(t,s-1);++r){const o=n[r],i=e[r+1];if(o>=0&&i>=0&&o!==1&&o!==i)throw new Error(`defaultValue.shape=${n}, and ragged tensor input flatValues.shape=${e} are incompatible: defaultValue.shape[${r-n.length}] = ${o} but ragged tensor input.flatValues.shape[${r-n.length}] = ${i}`)}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lr=30;function ss(n){return n<=lr?n:Ss(n,Math.floor(Math.sqrt(n)))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ji(n,e,t){const s=t*(typeof n=="number"?n:n[0]),r=e*(typeof n=="number"?n:n[1]);return[s,r]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ur(n,e,t,s=!0){let r=[];if(s)r=r.concat(e.slice(0)),r.push(n[0]/t),r=r.concat(n.slice(1));else{r=r.concat(n[0]);const o=e.length;for(let i=0;i<o;++i)r=r.concat([n[i+1]/e[i],e[i]]);r=r.concat(n.slice(o+1))}return r}function dr(n,e,t=!0){const s=[];if(t){s.push(e);for(let r=e+1;r<n;++r)r<=2*e?(s.push(r),s.push(r-(e+1))):s.push(r)}else{const r=[],o=[];for(let i=1;i<n;++i)i>=e*2+1||i%2===1?o.push(i):r.push(i);s.push(...r),s.push(0),s.push(...o)}return s}function hr(n,e,t,s=!0){const r=[];s?r.push(n[0]/t):r.push(n[0]*t);for(let o=1;o<n.length;++o)o<=e.length?s?r.push(e[o-1]*n[o]):r.push(n[o]/e[o-1]):r.push(n[o]);return r}function ea(n,e){const t=[0];for(let s=0;s<e;++s)t.push(n[s][0]);return t}function ta(n,e,t){const s=n.slice(0,1);for(let r=0;r<t;++r)s.push(n[r+1]-e[r][0]-e[r][1]);return s}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const na=1.7580993408473768,sa=1.0507009873554805;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ra=.3275911,oa=.254829592,ia=-.284496736,aa=1.421413741,ca=-1.453152027,la=1.061405429;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ms(n,e){if(n.length!==e.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${n.length}, imag: ${e.length}.`);const t=new Float32Array(n.length*2);for(let s=0;s<t.length;s+=2)t[s]=n[s/2],t[s+1]=e[s/2];return t}function cp(n){const e=new Float32Array(n.length/2),t=new Float32Array(n.length/2);for(let s=0;s<n.length;s+=2)e[s/2]=n[s],t[s/2]=n[s+1];return{real:e,imag:t}}function lp(n){const e=Math.ceil(n.length/4),t=new Float32Array(e),s=new Float32Array(e);for(let r=0;r<n.length;r+=4)t[Math.floor(r/4)]=n[r],s[Math.floor(r/4)]=n[r+1];return{real:t,imag:s}}function up(n){const e=Math.floor(n.length/4),t=new Float32Array(e),s=new Float32Array(e);for(let r=2;r<n.length;r+=4)t[Math.floor(r/4)]=n[r],s[Math.floor(r/4)]=n[r+1];return{real:t,imag:s}}function dp(n,e){const t=n[e*2],s=n[e*2+1];return{real:t,imag:s}}function hp(n,e,t,s){n[s*2]=e,n[s*2+1]=t}function fp(n,e){const t=new Float32Array(n/2),s=new Float32Array(n/2);for(let r=0;r<Math.ceil(n/2);r++){const o=(e?2:-2)*Math.PI*(r/n);t[r]=Math.cos(o),s[r]=Math.sin(o)}return{real:t,imag:s}}function pp(n,e,t){const s=(t?2:-2)*Math.PI*(n/e),r=Math.cos(s),o=Math.sin(s);return{real:r,imag:o}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ms="->",mp=/->/g,qr=",",Kr="...";function ua(n,e){n=n.replace(/\s/g,"");const t=(n.length-n.replace(mp,"").length)/ms.length;if(t<1)throw new Error("Equations without an arrow are not supported.");if(t>1)throw new Error(`Equation must contain exactly one arrow ("${ms}").`);const[s,r]=n.split(ms);I(s.indexOf(Kr)===-1,()=>`The ellipsis notation ("${Kr}") is not supported yet.`);const o=s.split(qr),i=o.length;if(e!==i)throw new Error(`Expected ${i} input tensors, received ${e}`);if(i>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const a=[];for(let h=0;h<r.length;++h){const f=r[h];if(!o.some(p=>p.indexOf(f)!==-1))throw new Error(`Output subscripts contain the label ${f} not present in the input subscripts.`);a.indexOf(f)===-1&&a.push(f)}for(let h=0;h<s.length;++h){const f=s[h];a.indexOf(f)===-1&&f!==qr&&a.push(f)}const c=new Array(o.length);for(let h=0;h<i;++h){if(new Set(o[h].split("")).size!==o[h].length)throw new Error(`Found duplicate axes in input component ${o[h]}. Support for duplicate axes in input is not implemented yet.`);c[h]=[];for(let f=0;f<o[h].length;++f)c[h].push(a.indexOf(o[h][f]))}const l=a.length,u=r.length,d=[];for(let h=u;h<l;++h)d.push(h);return{allDims:a,summedDims:d,idDims:c}}function da(n,e){let t=new Array(n);t.fill(-1);for(let r=0;r<e.length;++r)t[e[r]]=r;const s=[];for(let r=0;r<n;++r)t[r]===-1&&s.push(r);return t=t.filter(r=>r!==-1),{permutationIndices:t,expandDims:s}}function ha(n,e,t){const s=new Array(n);for(let r=0;r<t.length;++r){const o=t[r].shape;for(let i=0;i<e[r].length;++i)s[e[r][i]]===void 0?s[e[r][i]]=o[i]:I(s[e[r][i]]===o[i],()=>`Expected dimension ${s[e[r][i]]} at axis ${i} of input shaped ${JSON.stringify(o)}, but got dimension ${o[i]}`)}}function fa(n,e){const t=n,s=[];let r=0;n.length===0&&t.push(-1),r=n.length+1;for(let i=0;i<r;++i)s.push([]);const o=[];for(let i=0;i<t.length;++i){const a=t[i],c=gp(e,a);for(const l of c)o.indexOf(l)===-1&&(s[i].push(l),o.push(l))}return{path:t,steps:s}}function pa(n){return n.every((e,t)=>e===t)}function gp(n,e){const t=[];for(let s=0;s<n.length;++s)(n[s].length===0||n[s].indexOf(e)!==-1||e===-1)&&t.push(s);return t}function ma(n,e,t=0){let s=[];if(typeof e=="number")I(n.shape[t]%e===0,()=>"Number of splits must evenly divide the axis."),s=new Array(e).fill(n.shape[t]/e);else{const r=e.reduce((i,a)=>(a===-1&&(i+=1),i),0);I(r<=1,()=>"There should be only one negative value in split array.");const o=e.indexOf(-1);if(o!==-1){const i=e.reduce((a,c)=>c>0?a+c:a);e[o]=n.shape[t]-i}I(n.shape[t]===e.reduce((i,a)=>i+a),()=>"The sum of sizes must match the size of the axis dimension."),s=e}return s}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ga(n){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${n}`}function xa(n,e){return`indices(${n}, 0) is invalid: ${e} < 0`}function Ca(n,e,t){return`indices(${n}, 0) is invalid: ${e} >= ${t}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ba(n,e){return`only one output dimension may be -1, not both ${n} and ${e}`}function wa(n,e){return`size ${n} must be non-negative, not ${e}`}function ya(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function $a(n,e){const t=E(n),s=E(e);return`Input to reshape is a SparseTensor with ${t}
  dense values, but the requested shape requires a multiple of ${s}. inputShape=${n} outputShape= ${e}`}function va(n,e){const t=E(n),s=E(e);return`Input to reshape is a tensor with ${t} dense values, but the requested shape has ${s}. inputShape=${n} outputShape=${e}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vs(){return"segment ids must be >= 0"}function Sa(){return"segment ids are not increasing"}function Ia(n,e){return`Segment id ${n} out of range [0, ${e}), possibly because segmentIds input is not sorted.`}function Ra(n,e,t){return`Bad: indices[${n}] == ${e} out of range [0, ${t})`}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ta(n,e){let t=!1,s;for(n<=lr?(s=n,t=!0):s=Ss(n,Math.floor(Math.sqrt(n)));!t;)s>e||s===n?t=!0:s=Ss(n,s+1);return s}function Ea(n,e,t){const s=[],r=n.length;for(let o=0;o<r;o++)o!==e?s.push(n[o]):s.push(t);return s}function Na(n,e,t,s){const r=e.shape.length,o=n.shape.length;if(s!==0&&(s<-r||s>r))throw new Error(`Expect batchDims in the range of [-${r}, ${r}], but got ${s}`);if(s<0&&(s+=r),s>o)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${o}).`);if(t<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${t}).`);for(let d=0;d<s;++d)if(n.shape[d]!==e.shape[d])throw new Error(`x.shape[${d}]: ${n.shape[d]} should be equal to indices.shape[${d}]: ${e.shape[d]}.`);const i=n.shape[t],a=[];let c=1,l=1,u=1;for(let d=0;d<s;++d)a.push(n.shape[d]),c*=n.shape[d];for(let d=s;d<t;d++)a.push(n.shape[d]),l*=n.shape[d];for(let d=s;d<r;d++)a.push(e.shape[d]);for(let d=t+1;d<o;d++)a.push(n.shape[d]),u*=n.shape[d];return{batchSize:c,sliceSize:u,outerSize:l,dimSize:i,outputShape:a}}const xp=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:Na,computeOutShape:Ea,segOpComputeOptimalWindowSize:Ta},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gt(n){try{return n.map(e=>Vt(e))}catch(e){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function ka(n){return n.map(e=>ht(e))}const Cp=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:oa,ERF_A2:ia,ERF_A3:aa,ERF_A4:ca,ERF_A5:la,ERF_P:ra,PARALLELIZE_THRESHOLD:lr,get RowPartitionType(){return Oe},SELU_SCALE:sa,SELU_SCALEALPHA:na,applyActivation:yf,assertAndGetBroadcastShape:ae,assertAxesAreInnerMostDims:Me,assertParamsConsistent:qi,assignToTypedArray:hp,axesAreInnerMostDims:sr,calculateShapes:ns,checkEinsumDimSizes:ha,checkPadOnDimRoundingMode:Uh,combineLocations:Ai,combineRaggedTensorToTensorShapes:Ki,complexWithEvenIndex:lp,complexWithOddIndex:up,computeConv2DInfo:Be,computeConv3DInfo:wn,computeDefaultPad:tr,computeDilation2DInfo:Ni,computeOptimalWindowSize:ss,computeOutAndReduceShapes:He,computeOutShape:bt,computePool2DInfo:jt,computePool3DInfo:bn,convertConv2DDataFormat:Kt,decodeEinsumEquation:ua,eitherStridesOrDilationsAreOne:qt,expandShapeToKeepDim:je,exponent:pp,exponents:fp,fromStringArrayToUint8:ka,fromUint8ToStringArray:Gt,getAxesPermutation:Te,getBroadcastDims:Xn,getComplexWithIndex:dp,getEinsumComputePath:fa,getEinsumPermutation:da,getFusedBiasGradient:wf,getFusedDyActivation:bf,getImageCenter:Ji,getInnerMostAxes:Ee,getPermuted:dr,getRaggedRank:Qi,getReductionAxes:ki,getReshaped:ur,getReshapedPermuted:hr,getRowPartitionTypesHelper:Yi,getSliceBeginCoords:ea,getSliceSize:ta,getSparseFillEmptyRowsIndicesDenseShapeMismatch:ga,getSparseFillEmptyRowsNegativeIndexErrorMessage:xa,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:Ca,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:ya,getSparseReshapeInputOutputMismatchErrorMessage:va,getSparseReshapeInputOutputMultipleErrorMessage:$a,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:ba,getSparseReshapeNegativeOutputDimErrorMessage:wa,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:Ra,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:Vs,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:Sa,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:Ia,getUndoAxesPermutation:rr,isIdentityPermutation:pa,log:Ed,mergeRealAndImagArrays:Ms,prepareAndValidate:Pi,prepareSplitSize:ma,segment_util:xp,shouldFuse:$f,slice_util:op,splitRealAndImagArrays:cp,stridesOrDilationsArePositive:Vh,tupleValuesAreOne:_s,upcastType:ze,validateDefaultValueShape:Zi,validateInput:xf,validateUpdateShape:Fi,warn:Pe},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Gf();/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lt={},En={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function bp(n,e){lt[n]=e}function _e(n,e){if(!(n in lt)||e!=null){const s=yp(n,e);if(s!==null)lt[n]=s;else return console.log("Could not get context for WebGL version",n),null}const t=lt[n];return t==null||t.isContextLost()?(delete lt[n],_e(n)):(t.disable(t.DEPTH_TEST),t.disable(t.STENCIL_TEST),t.disable(t.BLEND),t.disable(t.DITHER),t.disable(t.POLYGON_OFFSET_FILL),t.disable(t.SAMPLE_COVERAGE),t.enable(t.SCISSOR_TEST),t.enable(t.CULL_FACE),t.cullFace(t.BACK),lt[n])}function wp(n){if(!y().getBool("IS_SAFARI")&&typeof OffscreenCanvas<"u"&&n===2)return new OffscreenCanvas(300,150);if(typeof document<"u")return document.createElement("canvas");throw new Error("Cannot create a canvas in this context")}function yp(n,e){if(n!==1&&n!==2)throw new Error("Cannot get WebGL rendering context, WebGL is disabled.");const t=e??wp(n);return t.addEventListener("webglcontextlost",s=>{s.preventDefault(),delete lt[n]},!1),y().getBool("SOFTWARE_WEBGL_ENABLED")&&(En.failIfMajorPerformanceCaveat=!1),n===1?t.getContext("webgl",En)||t.getContext("experimental-webgl",En):t.getContext("webgl2",En)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var fn;(function(n){n[n.DENSE=0]="DENSE",n[n.SHARED_BATCH=1]="SHARED_BATCH"})(fn||(fn={}));var Ce;(function(n){n[n.RENDER=0]="RENDER",n[n.UPLOAD=1]="UPLOAD",n[n.PIXELS=2]="PIXELS",n[n.DOWNLOAD=3]="DOWNLOAD"})(Ce||(Ce={}));var Q;(function(n){n[n.UNPACKED_FLOAT16=0]="UNPACKED_FLOAT16",n[n.UNPACKED_FLOAT32=1]="UNPACKED_FLOAT32",n[n.PACKED_4X1_UNSIGNED_BYTE=2]="PACKED_4X1_UNSIGNED_BYTE",n[n.PACKED_2X2_FLOAT32=3]="PACKED_2X2_FLOAT32",n[n.PACKED_2X2_FLOAT16=4]="PACKED_2X2_FLOAT16"})(Q||(Q={}));function yn(n,e){return[e,n]}function $p(n,e){return n*e}function Nn(n){const e=E(n),t=Math.ceil(e/4);return $s(t)}function Yt(n,e){return[Math.max(1,Math.ceil(e/2)),Math.max(1,Math.ceil(n/2))]}function vp(n,e){const[t,s]=Yt(n,e);return t*s*4}function fr(n,e){const t=n;let s,r,o,i,a,c,l,u,d,h;return y().getNumber("WEBGL_VERSION")===2?(s=t.R32F,r=t.R16F,o=t.RGBA16F,i=t.RGBA32F,a=t.RED,l=4,u=1,d=t.HALF_FLOAT,h=t.FLOAT,c=t.RGBA8):(s=n.RGBA,r=n.RGBA,o=n.RGBA,i=t.RGBA,a=n.RGBA,l=4,u=4,d=e!=null?e.HALF_FLOAT_OES:null,h=n.FLOAT,c=n.RGBA),{internalFormatFloat:s,internalFormatHalfFloat:r,internalFormatPackedHalfFloat:o,internalFormatPackedFloat:i,textureFormatFloat:a,downloadTextureFormat:c,downloadUnpackNumChannels:l,defaultNumChannels:u,textureTypeHalfFloat:d,textureTypeFloat:h}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function k(n,e){const t=e();return y().getBool("DEBUG")&&Sp(n),t}function Sp(n){const e=n.getError();if(e!==n.NO_ERROR)throw new Error("WebGL Error: "+Ep(n,e))}const Ip=596e-10,Rp=65504;function Tp(n){return!!(y().getBool("WEBGL_RENDER_FLOAT32_ENABLED")||n===0||Ip<Math.abs(n)&&Math.abs(n)<Rp)}function Ep(n,e){switch(e){case n.NO_ERROR:return"NO_ERROR";case n.INVALID_ENUM:return"INVALID_ENUM";case n.INVALID_VALUE:return"INVALID_VALUE";case n.INVALID_OPERATION:return"INVALID_OPERATION";case n.INVALID_FRAMEBUFFER_OPERATION:return"INVALID_FRAMEBUFFER_OPERATION";case n.OUT_OF_MEMORY:return"OUT_OF_MEMORY";case n.CONTEXT_LOST_WEBGL:return"CONTEXT_LOST_WEBGL";default:return`Unknown error code ${e}`}}function kn(n,e){return qe(n,()=>n.getExtension(e),'Extension "'+e+'" not supported on this browser.')}function Np(n,e){const t=qe(n,()=>n.createShader(n.VERTEX_SHADER),"Unable to create vertex WebGLShader.");if(k(n,()=>n.shaderSource(t,e)),k(n,()=>n.compileShader(t)),n.getShaderParameter(t,n.COMPILE_STATUS)===!1)throw console.log(n.getShaderInfoLog(t)),new Error("Failed to compile vertex shader.");return t}function kp(n,e){const t=qe(n,()=>n.createShader(n.FRAGMENT_SHADER),"Unable to create fragment WebGLShader.");if(k(n,()=>n.shaderSource(t,e)),k(n,()=>n.compileShader(t)),y().get("ENGINE_COMPILE_ONLY"))return t;if(n.getShaderParameter(t,n.COMPILE_STATUS)===!1)throw Aa(e,n.getShaderInfoLog(t)),new Error("Failed to compile fragment shader.");return t}const Ap=/ERROR: [0-9]+:([0-9]+):/g;function Aa(n,e){const t=Ap.exec(e);if(t==null){console.log(`Couldn't parse line number in error: ${e}`),console.log(n);return}const s=+t[1],r=n.split(`
`),o=r.length.toString().length+2,i=r.map((d,h)=>_t((h+1).toString(),o)+d);let a=0;for(let d=0;d<i.length;d++)a=Math.max(i[d].length,a);const c=i.slice(0,s-1),l=i.slice(s-1,s),u=i.slice(s);console.log(c.join(`
`)),console.log(e.split(`
`)[0]),console.log(`%c ${_t(l[0],a)}`,"border:1px solid red; background-color:#e3d2d2; color:#a61717"),console.log(u.join(`
`))}function Fp(n){return qe(n,()=>n.createProgram(),"Unable to create WebGLProgram.")}function Dp(n,e){if(k(n,()=>n.linkProgram(e)),!y().get("ENGINE_COMPILE_ONLY")&&n.getProgramParameter(e,n.LINK_STATUS)===!1)throw console.log(n.getProgramInfoLog(e)),new Error("Failed to link vertex and fragment shaders.")}function gs(n,e){if(k(n,()=>n.validateProgram(e)),n.getProgramParameter(e,n.VALIDATE_STATUS)===!1)throw console.log(n.getProgramInfoLog(e)),new Error("Shader program validation failed.")}function Op(n,e){const t=qe(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return k(n,()=>n.bindBuffer(n.ARRAY_BUFFER,t)),k(n,()=>n.bufferData(n.ARRAY_BUFFER,e,n.STATIC_DRAW)),t}function Pp(n,e){const t=qe(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return k(n,()=>n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,t)),k(n,()=>n.bufferData(n.ELEMENT_ARRAY_BUFFER,e,n.STATIC_DRAW)),t}function _p(n){return qe(n,()=>n.createTexture(),"Unable to create WebGLTexture.")}function Lp(n,e){const t=y().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(n<=0||e<=0){const s=`[${n}x${e}]`;throw new Error("Requested texture size "+s+" is invalid.")}if(n>t||e>t){const s=`[${n}x${e}]`,r=`[${t}x${t}]`;throw new Error("Requested texture size "+s+" greater than WebGL maximum on this browser / GPU "+r+".")}}function Bp(n){return qe(n,()=>n.createFramebuffer(),"Unable to create WebGLFramebuffer.")}function Yr(n,e,t,s,r,o,i){const a=n.getAttribLocation(e,t);return a===-1?!1:(k(n,()=>n.bindBuffer(n.ARRAY_BUFFER,s)),k(n,()=>n.vertexAttribPointer(a,r,n.FLOAT,!1,o,i)),k(n,()=>n.enableVertexAttribArray(a)),!0)}function Mp(n,e,t){zp(n,t),k(n,()=>n.activeTexture(n.TEXTURE0+t)),k(n,()=>n.bindTexture(n.TEXTURE_2D,e))}function Vp(n,e,t){return qe(n,()=>n.getUniformLocation(e,t),'uniform "'+t+'" not present in program.')}function Up(n,e,t){return n.getUniformLocation(e,t)}function Wp(n,e,t,s){k(n,()=>Mp(n,e,s)),k(n,()=>n.uniform1i(t,s))}function xs(n,e,t){k(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,t)),k(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,e,0))}function Qr(n,e){k(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,e)),k(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,null,0))}function An(n){const e=n.checkFramebufferStatus(n.FRAMEBUFFER);if(e!==n.FRAMEBUFFER_COMPLETE)throw new Error("Error binding framebuffer: "+Gp(n,e))}function Gp(n,e){switch(e){case n.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return"FRAMEBUFFER_INCOMPLETE_DIMENSIONS";case n.FRAMEBUFFER_UNSUPPORTED:return"FRAMEBUFFER_UNSUPPORTED";default:return`unknown error ${e}`}}function qe(n,e,t){const s=k(n,()=>e());if(s==null)throw new Error(t);return s}function zp(n,e){const t=n.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,s=e+n.TEXTURE0;if(s<n.TEXTURE0||s>t){const r=`[gl.TEXTURE0, gl.TEXTURE${t}]`;throw new Error(`textureUnit must be in ${r}.`)}}function zt(n,e=2){return E(n.slice(0,n.length-e))}function Ht(n){if(n.length===0)throw Error("Cannot get rows and columns of an empty shape array.");return[n.length>1?n[n.length-2]:1,n[n.length-1]]}function Fn(n){let e=[1,1,1];return n.length===0||n.length===1&&n[0]===1||(e=[zt(n),...Ht(n)]),e}function Hp(n,e=!1){let t=y().getNumber("WEBGL_MAX_TEXTURE_SIZE"),s=y().getNumber("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE");s===1/0&&y().getBool("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE")&&(s=t/2),e&&(t=t*2,s=s*2,n=n.map((a,c)=>c>=n.length-2?Hs(n[c]):n[c]),n.length===1&&(n=[2,n[0]])),n.length!==2&&(n=yt(n).newShape);let r=E(n),o=null;n.length<=1&&r<=t?o=[1,r]:n.length===2&&n[0]<=t&&n[1]<=t?o=n:n.length===3&&n[0]*n[1]<=t&&n[2]<=t?o=[n[0]*n[1],n[2]]:n.length===3&&n[0]<=t&&n[1]*n[2]<=t?o=[n[0],n[1]*n[2]]:n.length===4&&n[0]*n[1]*n[2]<=t&&n[3]<=t?o=[n[0]*n[1]*n[2],n[3]]:n.length===4&&n[0]<=t&&n[1]*n[2]*n[3]<=t&&(o=[n[0],n[1]*n[2]*n[3]]);const i=o!=null&&Math.max(...o)>s&&Math.min(...o)<=(e?2:1)&&Math.min(...o)>0;if(o==null||i)if(e){const a=zt(n);let c=2,l=2;n.length&&([c,l]=Ht(n)),r=a*(c/2)*(l/2),o=$s(r).map(u=>u*2)}else o=$s(r);return o}function Dn(n){return n%2===0}function jn(n,e){if(n=n.slice(-2),e=e.slice(-2),J(n,e)||!n.length||!e.length||n[0]===0||n[1]===0||e[0]===0||e[1]===0)return!0;if(n.length!==e.length){const t=n[n.length-1],s=e[e.length-1];if(t===s||Dn(t)&&Dn(s)&&(n[0]===1||e[0]===1))return!0}return n[1]===e[1]&&Dn(n[0])&&Dn(e[0])}let Cs,bs;function Xp(n){if(Cs==null){const e=_e(n);Cs=e.getParameter(e.MAX_TEXTURE_SIZE)}return Cs}function jp(n){if(bs==null){const e=_e(n);bs=e.getParameter(e.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,bs)}function qp(n){if(n===0)return 0;let e;const t=_e(n);return Ie(t,"EXT_disjoint_timer_query_webgl2")&&n===2?e=2:Ie(t,"EXT_disjoint_timer_query")?e=1:e=0,e}function Ie(n,e){return n.getExtension(e)!=null}function Zr(n){try{if(_e(n)!=null)return!0}catch(e){return console.log("Error when getting WebGL context: ",e),!1}return!1}function Kp(n){if(n===0)return!1;const e=_e(n);if(n===1){if(!Ie(e,"OES_texture_float"))return!1}else if(!Ie(e,"EXT_color_buffer_float"))return!1;return Us(e)}function Yp(n){if(n===0)return!1;const e=_e(n);if(n===1){if(!Ie(e,"OES_texture_float")||!Ie(e,"WEBGL_color_buffer_float"))return!1}else{if(Ie(e,"EXT_color_buffer_float"))return Us(e);const s="EXT_color_buffer_half_float";if(Ie(e,s)){const r=e.getExtension(s);return Qp(e,r)}return!1}return Us(e)}function Us(n){const e=fr(n),t=n.createTexture();n.bindTexture(n.TEXTURE_2D,t),n.texImage2D(n.TEXTURE_2D,0,e.internalFormatFloat,1,1,0,e.textureFormatFloat,e.textureTypeFloat,null);const o=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,o),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,t,0);const i=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(t),n.deleteFramebuffer(o),i}function Qp(n,e){const t=fr(n,e),s=n.createTexture();n.bindTexture(n.TEXTURE_2D,s),n.texImage2D(n.TEXTURE_2D,0,t.internalFormatHalfFloat,1,1,0,t.textureFormatFloat,t.textureTypeHalfFloat,null);const i=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,i),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,s,0);const a=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(s),n.deleteFramebuffer(i),a}function Zp(n){return n!==2?!1:_e(n).fenceSync!=null}function $n(n,e){Array.isArray(n)||(n=[n]),n.forEach(t=>{t!=null&&I(t.dtype!=="complex64",()=>`${e} does not support complex64 tensors in the WebGL backend.`)})}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const A=y();A.registerFlag("HAS_WEBGL",()=>A.getNumber("WEBGL_VERSION")>0);A.registerFlag("WEBGL_VERSION",()=>Zr(2)?2:Zr(1)?1:0);A.registerFlag("WEBGL_CHECK_NUMERICAL_PROBLEMS",()=>!1);A.registerFlag("WEBGL_BUFFER_SUPPORTED",()=>A.get("WEBGL_VERSION")===2);A.registerFlag("WEBGL_CPU_FORWARD",()=>!0);A.registerFlag("WEBGL_FORCE_F16_TEXTURES",()=>!1);A.registerFlag("WEBGL_PACK",()=>A.getBool("HAS_WEBGL"));A.registerFlag("WEBGL_PACK_NORMALIZATION",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_CLIP",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_DEPTHWISECONV",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_BINARY_OPERATIONS",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_UNARY_OPERATIONS",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_ARRAY_OPERATIONS",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_IMAGE_OPERATIONS",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_REDUCE",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_LAZILY_UNPACK",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_CONV_IM2COL",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_PACK_CONV2DTRANSPOSE",()=>A.getBool("WEBGL_PACK"));A.registerFlag("WEBGL_MAX_TEXTURE_SIZE",()=>Xp(A.getNumber("WEBGL_VERSION")));A.registerFlag("WEBGL_MAX_TEXTURES_IN_SHADER",()=>jp(A.getNumber("WEBGL_VERSION")));A.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION",()=>{const n=A.getNumber("WEBGL_VERSION");return n===0?0:qp(n)});A.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE",()=>A.getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0&&!Ci());A.registerFlag("WEBGL_RENDER_FLOAT32_CAPABLE",()=>Kp(A.getNumber("WEBGL_VERSION")));A.registerFlag("WEBGL_RENDER_FLOAT32_ENABLED",()=>A.getBool("WEBGL_FORCE_F16_TEXTURES")?!1:A.getBool("WEBGL_RENDER_FLOAT32_CAPABLE"));A.registerFlag("WEBGL_DOWNLOAD_FLOAT_ENABLED",()=>Yp(A.getNumber("WEBGL_VERSION")));A.registerFlag("WEBGL_FENCE_API_ENABLED",()=>Zp(A.getNumber("WEBGL_VERSION")));A.registerFlag("WEBGL_SIZE_UPLOAD_UNIFORM",()=>A.getBool("WEBGL_RENDER_FLOAT32_ENABLED")?4:0);A.registerFlag("WEBGL_DELETE_TEXTURE_THRESHOLD",()=>-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${n}.`)});A.registerFlag("WEBGL_FLUSH_THRESHOLD",()=>Ci()?1:-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${n}.`)});A.registerFlag("CPU_HANDOFF_SIZE_THRESHOLD",()=>128);A.registerFlag("WEBGL_USE_SHAPES_UNIFORMS",()=>!1);A.registerFlag("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e5);A.registerFlag("TOPK_K_CPU_HANDOFF_THRESHOLD",()=>128);A.registerFlag("WEBGL_EXP_CONV",()=>!1);A.registerFlag("SOFTWARE_WEBGL_ENABLED",()=>A.getBool("IS_TEST"));A.registerFlag("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE",()=>1/0);A.registerFlag("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE",()=>!1);A.registerFlag("WEBGL2_ISNAN_CUSTOM",()=>!1);A.registerFlag("ENGINE_COMPILE_ONLY",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function le(){let n,e,t,s,r,o,i,a,c,l;return y().getNumber("WEBGL_VERSION")===2?(n="#version 300 es",e="in",t="out",s="in",r="texture",o="outputColor",i="out vec4 outputColor;",a=y().getBool("WEBGL2_ISNAN_CUSTOM")?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:"",c="",l=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(n="",e="attribute",t="varying",s="varying",r="texture2D",o="gl_FragColor",i="",a=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,c=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,l=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:n,attribute:e,varyingVs:t,varyingFs:s,texture2D:r,output:o,defineOutput:i,defineSpecialNaN:a,defineSpecialInf:c,defineRound:l}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rt(n,e,t="index"){const s=Z(e);return s.map((r,o)=>{const i=`int ${n[o]} = ${t} / ${r}`,a=o===s.length-1?`int ${n[o+1]} = ${t} - ${n[o]} * ${r}`:`index -= ${n[o]} * ${r}`;return`${i}; ${a};`}).join("")}function rs(n,e,t="index"){const s=Z(e);return s.map((r,o)=>{const i=`int ${n[o]} = ${t} / outShapeStrides[${o}]`,a=o===s.length-1?`int ${n[o+1]} = ${t} - ${n[o]} * outShapeStrides[${o}]`:`index -= ${n[o]} * outShapeStrides[${o}]`;return`${i}; ${a};`}).join("")}function Jp(n,e){const t=n.length,s=n.map(o=>`${e}[${o}]`),r=new Array(t-1);r[t-2]=s[t-1];for(let o=t-3;o>=0;--o)r[o]=`(${r[o+1]} * ${s[o+1]})`;return r}function em(n,e,t="index"){const s=n.map((o,i)=>i),r=Jp(s,e);return r.map((o,i)=>{const a=`int ${n[i]} = ${t} / ${r[i]}`,c=i===r.length-1?`int ${n[i+1]} = ${t} - ${n[i]} * ${r[i]}`:`index -= ${n[i]} * ${r[i]}`;return`${a}; ${c};`}).join("")}function pr(n){const e=Z(n).map(t=>t.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${e[0]} + coords.y * ${e[1]} + coords.z;
  }
`}function mr(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}const Fa=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const{getBroadcastDims:Da}=Cp;function tm(n,e,t){const s=[];if(n.forEach(f=>{const p=E(f.shapeInfo.logicalShape);if(f.shapeInfo.isUniform?s.push(`uniform float ${f.name}${p>1?`[${p}]`:""};`):(s.push(`uniform sampler2D ${f.name};`),s.push(`uniform int offset${f.name};`)),t.enableShapeUniforms){const{uniformShape:x}=gr(t.packedInputs,f.shapeInfo.logicalShape,f.shapeInfo.texShape);switch(x.length){case 1:s.push(`uniform int ${f.name}Shape;`);break;case 2:s.push(`uniform ivec2 ${f.name}Shape;`);break;case 3:s.push(`uniform ivec3 ${f.name}Shape;`);break;case 4:s.push(`uniform ivec4 ${f.name}Shape;`);break}s.push(`uniform ivec2 ${f.name}TexShape;`)}}),t.enableShapeUniforms){switch(e.logicalShape.length){case 1:s.push("uniform int outShape;");break;case 2:s.push("uniform ivec2 outShape;"),s.push("uniform int outShapeStrides;");break;case 3:s.push("uniform ivec3 outShape;"),s.push("uniform ivec2 outShapeStrides;");break;case 4:s.push("uniform ivec4 outShape;"),s.push("uniform ivec3 outShapeStrides;");break}s.push("uniform ivec2 outTexShape;")}t.customUniforms&&t.customUniforms.forEach(f=>{s.push(`uniform ${f.type} ${f.name}${f.arrayIndex?`[${f.arrayIndex}]`:""};`)});const r=s.join(`
`),o=n.map(f=>nm(f,e,t.packedInputs,t.enableShapeUniforms)).join(`
`),i=e.texShape,a=le(),c=om(a);let l,u,d=cm(a);return e.isPacked?(l=sm(e.logicalShape,i,t.enableShapeUniforms),u=am(a)):(l=rm(e.logicalShape,i,t.enableShapeUniforms),u=im(a)),t.packedInputs&&(d+=hm),[d,c,u,r,l,o,t.userCode].join(`
`)}function Qt(n,e=!1){const t=n.shapeInfo.logicalShape;switch(t.length){case 0:return Sm(n,e);case 1:return Rm(n,e);case 2:return Em(n,e);case 3:return km(n,e);case 4:return Fm(n,e);case 5:return Dm(n);case 6:return Om(n);default:throw new Error(`${t.length}-D input sampling is not yet supported`)}}function Oa(n,e){switch(n.shapeInfo.logicalShape.length){case 0:return vm(n);case 1:return Im(n,e);case 2:return Tm(n,e);case 3:return Nm(n,e);default:return Am(n,e)}}function nm(n,e,t=!1,s){let r="";t?r+=Oa(n,s):r+=Qt(n,s);const o=n.shapeInfo.logicalShape,i=e.logicalShape;return o.length<=i.length&&(t?r+=Pm(n,e):r+=_m(n,e)),r}function sm(n,e,t){switch(n.length){case 0:return Pa();case 1:return fm(n,e,t);case 2:return ym(n,e,t);case 3:return mm(n,e,t);default:return xm(n,e,t)}}function rm(n,e,t){switch(n.length){case 0:return Pa();case 1:return pm(n,e,t);case 2:return $m(n,e,t);case 3:return gm(n,e,t);case 4:return Cm(n,e,t);case 5:return bm(n,e);case 6:return wm(n,e);default:throw new Error(`${n.length}-D output sampling is not yet supported`)}}function om(n){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${n.texture2D}(textureSampler, uv).r;
    }
  `}function im(n){return`
    void setOutput(float val) {
      ${n.output} = vec4(val, 0, 0, 0);
    }
  `}function am(n){return`
    void setOutput(vec4 val) {
      ${n.output} = val;
    }
  `}function cm(n){return`${n.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${n.varyingFs} vec2 resultUV;
    ${n.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${n.defineSpecialNaN}
    ${n.defineSpecialInf}
    ${n.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${lm}
    ${um}
    ${dm}
  `}const lm=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,um=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,dm=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,hm=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function Pa(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function fm(n,e,t){const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];return s[0]===1?t?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${s[1]}.0);
      }
    `:s[1]===1?t?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${s[0]}.0);
      }
    `:t?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      return 2 * (resTexRC.x * ${s[1]} + resTexRC.y);
    }
  `}function pm(n,e,t){return e[0]===1?t?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${e[1]}.0);
      }
    `:e[1]===1?t?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${e[0]}.0);
      }
    `:t?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      return resTexRC.x * ${e[1]} + resTexRC.y;
    }
  `}function mm(n,e,t){if(t)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],r=Math.ceil(n[2]/2),o=r*Math.ceil(n[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      int b = index / ${o};
      index -= b * ${o};

      int r = 2 * (index / ${r});
      int c = imod(index, ${r}) * 2;

      return ivec3(b, r, c);
    }
  `}function gm(n,e,t){if(t)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${rs(["r","c","d"],n)}
    return ivec3(r, c, d);
  }
`;const s=Rt(["r","c","d"],n);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      ${s}
      return ivec3(r, c, d);
    }
  `}function xm(n,e,t){if(t)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)],r=Math.ceil(n[n.length-1]/2),o=r*Math.ceil(n[n.length-2]/2);let i=o,a="",c="b, r, c";for(let l=2;l<n.length-1;l++)i*=n[n.length-l-1],a=`
      int b${l} = index / ${i};
      index -= b${l} * ${i};
    `+a,c=`b${l}, `+c;return`
    ivec${n.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      ${a}

      int b = index / ${o};
      index -= b * ${o};

      int r = 2 * (index / ${r});
      int c = imod(index, ${r}) * 2;

      return ivec${n.length}(${c});
    }
  `}function Cm(n,e,t){if(t)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${rs(["r","c","d","d2"],n)}
      return ivec4(r, c, d, d2);
    }
  `;const s=Rt(["r","c","d","d2"],n);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      ${s}
      return ivec4(r, c, d, d2);
    }
  `}function bm(n,e){const t=Rt(["r","c","d","d2","d3"],n);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${e[0]},
                             ${e[1]}));

      int index = resTexRC.x * ${e[1]} + resTexRC.y;

      ${t}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function wm(n,e){const t=Rt(["r","c","d","d2","d3","d4"],n);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;

      ${t}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function ym(n,e,t){const s=[Math.ceil(e[0]/2),Math.ceil(e[1]/2)];if(J(n,e))return t?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${s[0]}, ${s[1]}));
      }
    `;const r=Math.ceil(n[1]/2);return t?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));

      int index = resTexRC.x * ${s[1]} + resTexRC.y;
      int r = 2 * (index / ${r});
      int c = imod(index, ${r}) * 2;

      return ivec2(r, c);
    }
  `}function $m(n,e,t){return J(n,e)?t?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${e[0]}, ${e[1]}));
      }
    `:n[1]===1?t?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${e[0]}, ${e[1]}));
        int index = resTexRC.x * ${e[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:n[0]===1?t?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${e[0]}, ${e[1]}));
        int index = resTexRC.x * ${e[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:t?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${e[0]}, ${e[1]}));
      int index = resTexRC.x * ${e[1]} + resTexRC.y;
      int r = index / ${n[1]};
      int c = index - r * ${n[1]};
      return ivec2(r, c);
    }
  `}function Tt(n){return`offset${n}`}function vm(n){const e=n.name,t="get"+e.charAt(0).toUpperCase()+e.slice(1),s=le();return`
    vec4 ${t}() {
      return ${s.texture2D}(${e}, halfCR);
    }
  `}function Sm(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1);if(n.shapeInfo.isUniform)return`float ${s}() {return ${t};}`;const[r,o]=n.shapeInfo.texShape;if(r===1&&o===1)return`
      float ${s}() {
        return sampleTexture(${t}, halfCR);
      }
    `;const i=Tt(t);if(e)return`
    float ${s}() {
      vec2 uv = uvFromFlat(${t}TexShape[0], ${t}TexShape[1], ${i});
      return sampleTexture(${t}, uv);
    }
  `;const[a,c]=n.shapeInfo.texShape;return`
    float ${s}() {
      vec2 uv = uvFromFlat(${a}, ${c}, ${i});
      return sampleTexture(${t}, uv);
    }
  `}function Im(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),r=n.shapeInfo.texShape,o=le();if(e)return`
    vec4 ${s}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${t}TexShape[0]) / 2.0), ceil(float(${t}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${o.texture2D}(${t}, uv);
    }
  `;const i=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)];return`
    vec4 ${s}(int index) {
      vec2 uv = packedUVfrom1D(
        ${i[0]}, ${i[1]}, index);
      return ${o.texture2D}(${t}, uv);
    }
  `}function Rm(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1);if(n.shapeInfo.isUniform)return`
      float ${s}(int index) {
        ${Zt(n)}
      }
    `;const r=n.shapeInfo.texShape,o=r[0],i=r[1];if(i===1&&o===1)return`
      float ${s}(int index) {
        return sampleTexture(${t}, halfCR);
      }
    `;const a=Tt(t);return i===1?e?`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${a}) + 0.5) / float(${t}TexShape[0]));
        return sampleTexture(${t}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${a}) + 0.5) / ${o}.0);
        return sampleTexture(${t}, uv);
      }
    `:o===1?e?`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${a}) + 0.5) / float(${t}TexShape[1]), 0.5);
        return sampleTexture(${t}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${a}) + 0.5) / ${i}.0, 0.5);
        return sampleTexture(${t}, uv);
      }
    `:e?`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${t}TexShape[0], ${t}TexShape[1], index + ${a});
      return sampleTexture(${t}, uv);
    }
  `:`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${o}, ${i}, index + ${a});
      return sampleTexture(${t}, uv);
    }
  `}function Tm(n,e){const t=n.shapeInfo.logicalShape,s=n.name,r="get"+s.charAt(0).toUpperCase()+s.slice(1),o=n.shapeInfo.texShape,i=o[0],a=o[1],c=le();if(o!=null&&J(t,o))return e?`
      vec4 ${r}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);

        return ${c.texture2D}(${s}, uv);
      }
    `:`
      vec4 ${r}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${a}.0, ${i}.0);

        return ${c.texture2D}(${s}, uv);
      }
    `;if(e)return`
    vec4 ${r}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${c.texture2D}(${s}, uv);
    }
  `;const l=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)],u=Math.ceil(t[1]/2);return`
    vec4 ${r}(int row, int col) {
      vec2 uv = packedUVfrom2D(${u}, ${l[0]}, ${l[1]}, row, col);
      return ${c.texture2D}(${s}, uv);
    }
  `}function Em(n,e){const t=n.shapeInfo.logicalShape,s=n.name,r="get"+s.charAt(0).toUpperCase()+s.slice(1),o=n.shapeInfo.texShape;if(o!=null&&J(t,o)){if(e)return`
      float ${r}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `;const h=o[0],f=o[1];return`
    float ${r}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${f}.0, ${h}.0);
      return sampleTexture(${s}, uv);
    }
  `}const{newShape:i,keptDims:a}=yt(t),c=i;if(c.length<t.length){const h=Jt(n,c),f=["row","col"];return`
      ${Qt(h,e)}
      float ${r}(int row, int col) {
        return ${r}(${en(f,a)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${r}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${t[1]}, 1)));
        ${Zt(n)}
      }
    `;const l=o[0],u=o[1],d=Tt(s);return u===1?e?`
      float ${r}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${s}TexShape[0]));
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${r}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${t[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${l}.0);
      return sampleTexture(${s}, uv);
    }
  `:l===1?e?`
      float ${r}(int row, int col) {
        float index = dot(vec3(row, col, ${d}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${s}TexShape[1]), 0.5);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${r}(int row, int col) {
      float index = dot(vec3(row, col, ${d}), vec3(${t[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${s}, uv);
    }
  `:e?`
      float ${r}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${s}Shape[1] + col + ${d};
        vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
        return sampleTexture(${s}, uv);
      }
    `:`
  float ${r}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${t[1]} + col + ${d};
    vec2 uv = uvFromFlat(${l}, ${u}, index);
    return sampleTexture(${s}, uv);
  }
`}function Nm(n,e){const t=n.shapeInfo.logicalShape,s=n.name,r="get"+s.charAt(0).toUpperCase()+s.slice(1),o=n.shapeInfo.texShape,i=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)];if(t[0]===1){const h=t.slice(1),f=[1,2],p=Jt(n,h),x=["b","row","col"];return`
        ${Oa(p,e)}
        vec4 ${r}(int b, int row, int col) {
          return ${r}(${en(x,f)});
        }
      `}const a=le();if(e)return`
    vec4 ${r}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${a.texture2D}(${s}, uv);
    }
  `;const c=i[0],l=i[1],u=Math.ceil(t[2]/2),d=u*Math.ceil(t[1]/2);return`
    vec4 ${r}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${c}, ${l}, ${d}, ${u}, b, row, col);
      return ${a.texture2D}(${s}, uv);
    }
  `}function km(n,e){const t=n.shapeInfo.logicalShape,s=n.name,r="get"+s.charAt(0).toUpperCase()+s.slice(1),o=t[1]*t[2],i=t[2],{newShape:a,keptDims:c}=yt(t),l=a;if(l.length<t.length){const x=Jt(n,l),g=["row","col","depth"];return`
        ${Qt(x,e)}
        float ${r}(int row, int col, int depth) {
          return ${r}(${en(g,c)});
        }
      `}if(n.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${o}, ${i}, 1)));
        ${Zt(n)}
      }
    `;const u=n.shapeInfo.texShape,d=u[0],h=u[1],f=n.shapeInfo.flatOffset;if(h===o&&f==null)return e?`
      float ${r}(int row, int col, int depth) {
        int stride1 = ${s}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
        float ${r}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${i}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${h}.0, ${d}.0);
          return sampleTexture(${s}, uv);
        }
      `;if(h===i&&f==null)return e?`
      float ${r}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${s}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${r}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${t[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${h}.0, ${d}.0);
      return sampleTexture(${s}, uv);
    }
  `;const p=Tt(s);return e?`
    float ${r}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${s}Shape[1] * ${s}Shape[2];
      int stride1 = ${s}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${p};
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
      return sampleTexture(${s}, uv);
    }
    `:`
      float ${r}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${o} + col * ${i} + depth + ${p};
        vec2 uv = uvFromFlat(${d}, ${h}, index);
        return sampleTexture(${s}, uv);
      }
  `}function Am(n,e){const t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),r=le();if(e)return`
    vec4 ${s}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${t}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${t}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${t}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${t}TexShape[0]) / 2.0), ceil(float(${t}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${r.texture2D}(${t}, uv);
    }
  `;const o=n.shapeInfo.logicalShape,i=o.length,a=n.shapeInfo.texShape,c=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)],l=c[0],u=c[1],d=Math.ceil(o[i-1]/2);let h=d*Math.ceil(o[i-2]/2),f="int b, int row, int col",p=`b * ${h} + (row / 2) * ${d} + (col / 2)`;for(let x=2;x<i-1;x++)f=`int b${x}, `+f,h*=o[i-x-1],p=`b${x} * ${h} + `+p;return`
    vec4 ${s}(${f}) {
      int index = ${p};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${l});
      return ${r.texture2D}(${t}, uv);
    }
  `}function Fm(n,e){const t=n.shapeInfo.logicalShape,s=n.name,r="get"+s.charAt(0).toUpperCase()+s.slice(1),o=t[3],i=t[2]*o,a=t[1]*i,{newShape:c,keptDims:l}=yt(t);if(c.length<t.length){const C=Jt(n,c),b=["row","col","depth","depth2"];return`
      ${Qt(C,e)}
      float ${r}(int row, int col, int depth, int depth2) {
        return ${r}(${en(b,l)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${r}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${a}, ${i}, ${o}, 1)));
        ${Zt(n)}
      }
    `;const u=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,h=d[0],f=d[1],p=`int stride2 = ${s}Shape[3];`,x=`int stride1 = ${s}Shape[2] * stride2;`,g=`int stride0 = ${s}Shape[1] * stride1;`;if(f===a&&u==null)return e?`
      float ${r}(int row, int col, int depth, int depth2) {
        ${p}
        ${x}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${r}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${i}, ${o}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${h}.0);
        return sampleTexture(${s}, uv);
      }
    `;if(f===o&&u==null)return e?`
      float ${r}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${s}Shape[1] * ${s}Shape[2], ${s}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${r}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${t[1]*t[2]}, ${t[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${h}.0);
        return sampleTexture(${s}, uv);
      }
    `;const m=Tt(s);return e?`
    float ${r}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${p}
      ${x}
      ${g}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index + ${m});
      return sampleTexture(${s}, uv);
    }
  `:`
    float ${r}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${a} + col * ${i} +
          depth * ${o} + depth2;
      vec2 uv = uvFromFlat(${h}, ${f}, index + ${m});
      return sampleTexture(${s}, uv);
    }
  `}function Dm(n){const e=n.shapeInfo.logicalShape,t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),r=e[4],o=e[3]*r,i=e[2]*o,a=e[1]*i,{newShape:c,keptDims:l}=yt(e);if(c.length<e.length){const x=Jt(n,c),g=["row","col","depth","depth2","depth3"];return`
      ${Qt(x)}
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        return ${s}(${en(g,l)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${a}, ${i}, ${o}, ${r})) +
          depth3;
        ${Zt(n)}
      }
    `;const u=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,h=d[0],f=d[1];if(f===a&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${i}, ${o}, ${r}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${h}.0);
        return sampleTexture(${t}, uv);
      }
    `;if(f===r&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${e[1]*e[2]*e[3]},
               ${e[2]*e[3]}, ${e[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${h}.0);
        return sampleTexture(${t}, uv);
      }
    `;const p=Tt(t);return`
    float ${s}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${a} + col * ${i} + depth * ${o} +
          depth2 * ${r} + depth3 + ${p};
      vec2 uv = uvFromFlat(${h}, ${f}, index);
      return sampleTexture(${t}, uv);
    }
  `}function Om(n){const e=n.shapeInfo.logicalShape,t=n.name,s="get"+t.charAt(0).toUpperCase()+t.slice(1),{newShape:r,keptDims:o}=yt(e);if(r.length<e.length){const g=Jt(n,r),m=["row","col","depth","depth2","depth3","depth4"];return`
      ${Qt(g)}
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${s}(${en(m,o)});
      }
    `}const i=e[5],a=e[4]*i,c=e[3]*a,l=e[2]*c,u=e[1]*l;if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${l}, ${c}, ${a})) +
          dot(
            vec2(depth3, depth4),
            vec2(${i}, 1)));
        ${Zt(n)}
      }
    `;const d=n.shapeInfo.flatOffset,h=n.shapeInfo.texShape,f=h[0],p=h[1];if(p===u&&d==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${l}, ${c}, ${a}, ${i})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${f}.0);
        return sampleTexture(${t}, uv);
      }
    `;if(p===i&&d==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${e[1]*e[2]*e[3]*e[4]},
               ${e[2]*e[3]*e[4]},
               ${e[3]*e[4]},
               ${e[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${f}.0);
        return sampleTexture(${t}, uv);
      }
    `;const x=Tt(t);return`
    float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${l} + depth * ${c} +
          depth2 * ${a} + depth3 * ${i} + depth4 + ${x};
      vec2 uv = uvFromFlat(${f}, ${p}, index);
      return sampleTexture(${t}, uv);
    }
  `}function Zt(n){const e=n.name,t=E(n.shapeInfo.logicalShape);return t<2?`return ${e};`:`
    for (int i = 0; i < ${t}; i++) {
      if (i == index) {
        return ${e}[i];
      }
    }
  `}function Pm(n,e){const t=n.name,s=t.charAt(0).toUpperCase()+t.slice(1),r="get"+s+"AtOutCoords",o=n.shapeInfo.logicalShape.length,i=e.logicalShape.length,a=Da(n.shapeInfo.logicalShape,e.logicalShape),c=U(i),l=i-o;let u;const d=["x","y","z","w","u","v"];o===0?u="":i<2&&a.length>=1?u="coords = 0;":u=a.map(C=>`coords.${d[C+l]} = 0;`).join(`
`);let h="";i<2&&o>0?h="coords":h=n.shapeInfo.logicalShape.map((C,b)=>`coords.${d[b+l]}`).join(", ");let f="return outputValue;";const x=E(n.shapeInfo.logicalShape)===1,m=E(e.logicalShape)===1;if(o===1&&!x&&!m)f=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(x&&!m)i===1?f=`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:f=`
        return vec4(outputValue.x);
      `;else if(a.length){const C=o-2,b=o-1;a.indexOf(C)>-1&&a.indexOf(b)>-1?f="return vec4(outputValue.x);":a.indexOf(C)>-1?f="return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);":a.indexOf(b)>-1&&(f="return vec4(outputValue.xx, outputValue.zz);")}return`
    vec4 ${r}() {
      ${c} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${s}(${h});
      ${f}
    }
  `}function _m(n,e){const t=n.name,s=t.charAt(0).toUpperCase()+t.slice(1),r="get"+s+"AtOutCoords",o=e.texShape,i=n.shapeInfo.texShape,a=n.shapeInfo.logicalShape.length,c=e.logicalShape.length;if(!n.shapeInfo.isUniform&&a===c&&n.shapeInfo.flatOffset==null&&J(i,o))return`
      float ${r}() {
        return sampleTexture(${t}, resultUV);
      }
    `;const l=U(c),u=Da(n.shapeInfo.logicalShape,e.logicalShape),d=c-a;let h;const f=["x","y","z","w","u","v"];a===0?h="":c<2&&u.length>=1?h="coords = 0;":h=u.map(x=>`coords.${f[x+d]} = 0;`).join(`
`);let p="";return c<2&&a>0?p="coords":p=n.shapeInfo.logicalShape.map((x,g)=>`coords.${f[g+d]}`).join(", "),`
    float ${r}() {
      ${l} coords = getOutputCoords();
      ${h}
      return get${s}(${p});
    }
  `}function U(n){if(n<=1)return"int";if(n===2)return"ivec2";if(n===3)return"ivec3";if(n===4)return"ivec4";if(n===5)return"ivec5";if(n===6)return"ivec6";throw Error(`GPU for rank ${n} is not yet supported`)}function gr(n,e,t){const{newShape:s,keptDims:r}=yt(e),o=e.length,i=n&&o===3&&e[0]===1,a=i?e.slice(1):s,c=!n&&o>1&&!J(e,t)&&s.length<o||i;return{useSqueezeShape:c,uniformShape:c?a:e,keptDims:r}}function Jt(n,e){const t=JSON.parse(JSON.stringify(n));return t.shapeInfo.logicalShape=e,t}function en(n,e){return e.map(t=>n[t]).join(", ")}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lm(n,e,t,s){const r=t.map((u,d)=>{const h={logicalShape:u.shape,texShape:u.isUniform?null:u.texData.texShape,isUniform:u.isUniform,isPacked:u.isUniform?!1:u.texData.isPacked,flatOffset:null};return u.texData!=null&&u.texData.slice!=null&&u.texData.slice.flatOffset>0&&(h.flatOffset=u.texData.slice.flatOffset),{name:e.variableNames[d],shapeInfo:h}}),o=r.map(u=>u.shapeInfo),i={logicalShape:s.shape,texShape:s.texData.texShape,isUniform:!1,isPacked:s.texData.isPacked,flatOffset:null},a=tm(r,i,e),c=kp(n.gl,a),l=n.createProgram(c);return y().get("ENGINE_COMPILE_ONLY")?{program:e,fragmentShader:c,source:a,webGLProgram:l,inShapeInfos:o,outShapeInfo:i,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(n.buildVao(l),Object.assign({program:e,fragmentShader:c,source:a,webGLProgram:l,inShapeInfos:o,outShapeInfo:i},_a(n,e,l)))}function _a(n,e,t){const s=[],r=[];let o,i,a,c=null,l=null;l=n.getUniformLocation(t,"NAN",!1),y().getNumber("WEBGL_VERSION")===1&&(c=n.getUniformLocation(t,"INFINITY",!1));const u=!1;for(const d of e.variableNames){const h={name:d,uniform:n.getUniformLocation(t,d,u),offset:n.getUniformLocation(t,`offset${d}`,u)};e.enableShapeUniforms&&(h.shape=n.getUniformLocation(t,`${d}Shape`,u),h.texShape=n.getUniformLocation(t,`${d}TexShape`,u)),s.push(h)}if(e.enableShapeUniforms&&(o=n.getUniformLocation(t,"outShape",u),a=n.getUniformLocation(t,"outShapeStrides",u),i=n.getUniformLocation(t,"outTexShape",u)),e.customUniforms)for(const d of e.customUniforms)r.push(n.getUniformLocation(t,d.name,u));return{variablesLocations:s,customUniformLocations:r,infLoc:c,nanLoc:l,outShapeLocation:o,outShapeStridesLocation:a,outTexShapeLocation:i}}function Jr(n,e){if(n.length!==e.length)throw Error(`Binary was compiled with ${n.length} inputs, but was executed with ${e.length} inputs`);n.forEach((t,s)=>{const r=t.logicalShape,o=e[s],i=o.shape;if(!J(r,i))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${r} and ${i} must match`);if(t.isUniform&&o.isUniform)return;const a=t.texShape,c=o.isUniform?null:o.texData.texShape;if(!J(a,c))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${a} and ${c} must match`)})}function Bm(n,e,t,s,r){e.program.enableShapeUniforms||(Jr(e.inShapeInfos,t),Jr([e.outShapeInfo],[s]));const o=s.texData.texture,i=s.texData.texShape;s.texData.isPacked?n.setOutputPackedMatrixTexture(o.texture,i[0],i[1]):n.setOutputMatrixTexture(o.texture,i[0],i[1]),n.setProgram(e.webGLProgram),n.bindVertexArray(e.webGLProgram.vao),y().getNumber("WEBGL_VERSION")===1&&e.infLoc!==null&&n.gl.uniform1f(e.infLoc,1/0),e.nanLoc!==null&&n.gl.uniform1f(e.nanLoc,NaN);for(let c=0;c<t.length;++c){const l=t[c],{uniform:u,offset:d,shape:h,texShape:f}=e.variablesLocations[c];if(h){const{uniformShape:p}=gr(e.program.packedInputs,l.shape,l.texData.texShape);switch(p.length){case 1:n.gl.uniform1iv(h,new Int32Array(p));break;case 2:n.gl.uniform2iv(h,new Int32Array(p));break;case 3:n.gl.uniform3iv(h,new Int32Array(p));break;case 4:n.gl.uniform4iv(h,new Int32Array(p));break}}if(f&&n.gl.uniform2i(f,l.texData.texShape[0],l.texData.texShape[1]),u!=null){if(l.isUniform){if(E(l.shape)<2)n.gl.uniform1f(u,l.uniformValues[0]);else{let p=l.uniformValues;p instanceof Float32Array||(p=new Float32Array(p)),n.gl.uniform1fv(u,p)}continue}l.texData.slice!=null&&d!=null&&n.gl.uniform1i(d,l.texData.slice.flatOffset),n.setInputMatrixTexture(l.texData.texture.texture,u,c)}}const a=e.outShapeLocation;if(a)switch(s.shape.length){case 1:n.gl.uniform1iv(a,new Int32Array(s.shape));break;case 2:n.gl.uniform2iv(a,new Int32Array(s.shape));break;case 3:n.gl.uniform3iv(a,new Int32Array(s.shape));break;case 4:n.gl.uniform4iv(a,new Int32Array(s.shape));break}if(e.outShapeStridesLocation){const c=Z(s.shape);switch(s.shape.length){case 2:n.gl.uniform1iv(e.outShapeStridesLocation,new Int32Array(c));break;case 3:n.gl.uniform2iv(e.outShapeStridesLocation,new Int32Array(c));break;case 4:n.gl.uniform3iv(e.outShapeStridesLocation,new Int32Array(c));break}}if(e.outTexShapeLocation&&n.gl.uniform2i(e.outTexShapeLocation,s.texData.texShape[0],s.texData.texShape[1]),e.program.customUniforms&&r)for(let c=0;c<e.program.customUniforms.length;++c){const l=e.program.customUniforms[c],u=e.customUniformLocations[c],d=r[c];if(l.type==="float")n.gl.uniform1fv(u,d);else if(l.type==="vec2")n.gl.uniform2fv(u,d);else if(l.type==="vec3")n.gl.uniform3fv(u,d);else if(l.type==="vec4")n.gl.uniform4fv(u,d);else if(l.type==="int")n.gl.uniform1iv(u,d);else if(l.type==="ivec2")n.gl.uniform2iv(u,d);else if(l.type==="ivec3")n.gl.uniform3iv(u,d);else if(l.type==="ivec4")n.gl.uniform4iv(u,d);else throw Error(`uniform type ${l.type} is not supported yet.`)}n.executeProgram()}function Mm(n,e,t){let s="";e.concat(t).forEach(i=>{const a=i.texData!=null&&i.texData.slice!=null&&i.texData.slice.flatOffset>0;if(n.enableShapeUniforms&&!i.isUniform){const c=i.texData.texShape,{useSqueezeShape:l,uniformShape:u,keptDims:d}=gr(n.packedInputs,i.shape,c);let h="",f="",p="";if(u.length===1&&n.packedInputs){const $=[Math.ceil(c[0]/2),Math.ceil(c[1]/2)];h=`${$[0]>1}_${$[1]>1}`}else if(u.length===2&&!n.packedInputs)f=`${u[0]>1}_${u[1]>1}`;else if(u.length>2&&!n.packedInputs){const $=Z(u);p=`${$[0]===c[1]}_${$[$.length-1]===c[1]}`}const x=i.shape.length,g=u.length===2&&J(i.shape,c),m=E(i.shape)===1,C=Xn(i.shape,t.shape),b=!n.packedInputs&&x===t.shape.length&&J(c,t.texData.texShape),w=n.packedInputs||u.length>2?"":`${c[0]>1}_${c[1]>1}`;s+=`${x}_${b}_${l?d:""}_${u.length}_${m}_${C}_${g}_${h}_${f}_${p}_${w}_${a}`}else{const c=i.isUniform?"uniform":i.texData.texShape;s+=`${i.shape}_${c}_${a}`}});const r=n.userCode;let o=n.constructor.name;return o+="_"+s+"_"+r+`${y().getNumber("WEBGL_VERSION")}`,o}function se(n){return y().getBool("WEBGL_USE_SHAPES_UNIFORMS")&&n<=4}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Vm{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=fn.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=le();this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?rs(["r","c","d"],e):Rt(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${t.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Um{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=fn.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const t=le();this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?rs(["r","c","d"],e):Rt(["r","c","d"],e)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${t.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Wm{constructor(e){this.variableNames=["A"],this.outTexUsage=Ce.DOWNLOAD;const t=le();this.outputShape=e,this.userCode=`
      ${Fa}

      void main() {
        float x = getAAtOutCoords();
        ${t.output} = encode_float(x);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Gm{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=Ce.DOWNLOAD;const t=le();this.outputShape=e,this.userCode=`
      ${Fa}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${t.output} = encode_float(x);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zm={R:0,G:1,B:2,A:3};class eo{constructor(e,t=!1,s="RGBA"){this.variableNames=["A"],this.customUniforms=[{name:"texShape",type:"ivec2"}];const r=le();this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length);let o="result";t&&(o="floor(result * 255. + 0.5)");let i="";for(let a=0;a<s.length;a++){const c=s[a];i+=`
          if(offset == ${a}) {
            result = values[${zm[c]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?mr():pr(e)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${s.length});

        flatIndex = idiv(flatIndex, ${s.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${r.texture2D}(A, uv);
          ${i}
        }
        ${r.output} = vec4(${o}, 0., 0., 0.);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Hm{constructor(e,t=!1){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:"texShape",type:"ivec2"}];const s=le();this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length);let r="",o="result";t&&(o="floor(result * 255. + 0.5)");for(let i=0;i<=1;i++)for(let a=0;a<=1;a++){const c=i*2+a;r+=`
          localCoords = coords;
          if(localCoords[2] + ${a} < ${this.enableShapeUniforms?"outShape[2]":`${e[2]}`}) {
          localCoords[2] += ${a};
          if (localCoords[1] + ${i} < ${this.enableShapeUniforms?"outShape[1]":`${e[1]}`}) {
            localCoords[1] += ${i};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${s.texture2D}(A, uv);

            if (offset == 0) {
              result[${c}] = values[0];
            } else if (offset == 1) {
              result[${c}] = values[1];
            } else if (offset == 2) {
              result[${c}] = values[2];
            } else {
              result[${c}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?mr():pr(e)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${r}

          ${s.output} = ${o};
        }
    `}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xm(n){const e=le(),t=`${e.version}
    precision highp float;
    ${e.attribute} vec3 clipSpacePos;
    ${e.attribute} vec2 uv;
    ${e.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`;return Np(n,t)}function jm(n){const e=new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]);return Op(n,e)}function qm(n){const e=new Uint16Array([0,1,2,2,1,3]);return Pp(n,e)}function vn(n,e,t,s,r,o){Lp(e,t);const i=_p(n),a=n.TEXTURE_2D;return k(n,()=>n.bindTexture(a,i)),k(n,()=>n.texParameteri(a,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE)),k(n,()=>n.texParameteri(a,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE)),k(n,()=>n.texParameteri(a,n.TEXTURE_MIN_FILTER,n.NEAREST)),k(n,()=>n.texParameteri(a,n.TEXTURE_MAG_FILTER,n.NEAREST)),y().getNumber("WEBGL_VERSION")===1?k(n,()=>n.texImage2D(a,0,s,e,t,0,r,o,null)):k(n,()=>n.texStorage2D(a,1,s,e,t)),k(n,()=>n.bindTexture(n.TEXTURE_2D,null)),{texture:i,texShape:[t,e]}}function La(n){return n.internalFormatFloat}function Km(n,e,t,s){const[r,o]=yn(e,t);return vn(n,r,o,La(s),s.textureFormatFloat,n.FLOAT)}function Ba(n){return n.internalFormatHalfFloat}function Ym(n,e,t,s){const[r,o]=yn(e,t);return vn(n,r,o,Ba(s),s.textureFormatFloat,s.textureTypeHalfFloat)}function Ma(n){return n.downloadTextureFormat}function Qm(n,e,t,s){const[r,o]=yn(e,t);return vn(n,r,o,Ma(s),n.RGBA,n.UNSIGNED_BYTE)}function Va(n){return n.internalFormatPackedFloat}function Zm(n,e,t,s){const[r,o]=Yt(e,t);return vn(n,r,o,Va(s),n.RGBA,n.FLOAT)}function Ua(n){return n.internalFormatPackedHalfFloat}function Jm(n,e,t,s){const[r,o]=Yt(e,t);return vn(n,r,o,Ua(s),n.RGBA,s.textureTypeHalfFloat)}function eg(n,e,t){return k(n,()=>n.bindBuffer(n.ARRAY_BUFFER,t)),Yr(n,e,"clipSpacePos",t,3,20,0)&&Yr(n,e,"uv",t,2,20,12)}function tg(n,e,t,s,r,o){k(n,()=>n.bindTexture(n.TEXTURE_2D,e));let i,a,c;r instanceof Uint8Array?(i=new Uint8Array(t*s*4),a=n.UNSIGNED_BYTE,c=n.RGBA):(i=new Float32Array(t*s*4),a=n.FLOAT,c=o.internalFormatPackedFloat),i.set(r),y().getNumber("WEBGL_VERSION")===2?k(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,t,s,n.RGBA,a,i)):k(n,()=>n.texImage2D(n.TEXTURE_2D,0,c,t,s,0,n.RGBA,a,i)),k(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function ng(n,e,t){k(n,()=>n.bindTexture(n.TEXTURE_2D,e)),t.data instanceof Uint8Array?y().getNumber("WEBGL_VERSION")===2?k(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,t.width,t.height,n.RGBA,n.UNSIGNED_BYTE,t.data)):k(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,t.width,t.height,0,n.RGBA,n.UNSIGNED_BYTE,t.data)):y().getNumber("WEBGL_VERSION")===2?k(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,n.RGBA,n.UNSIGNED_BYTE,t)):k(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,n.RGBA,n.UNSIGNED_BYTE,t)),k(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function sg(n,e,t,s){const r=n.createBuffer();k(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,r));const a=4*4*e*t;return k(n,()=>n.bufferData(n.PIXEL_PACK_BUFFER,a,n.STREAM_READ)),k(n,()=>n.readPixels(0,0,t,e,n.RGBA,n.FLOAT,0)),k(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,null)),r}function rg(n,e,t){const s=n,r=new Float32Array(t);return s.bindBuffer(s.PIXEL_PACK_BUFFER,e),s.getBufferSubData(s.PIXEL_PACK_BUFFER,0,r),s.bindBuffer(s.PIXEL_PACK_BUFFER,null),r}function og(n,e,t,s){const[r,o]=yn(e,t),i=4,a=new Uint8Array($p(e*t,i));return k(n,()=>n.readPixels(0,0,r,o,s.downloadTextureFormat,n.UNSIGNED_BYTE,a)),new Float32Array(a.buffer)}function ig(n,e,t,s,r,o,i,a){const c=n,l=new Float32Array(vp(o,i));return c.bindBuffer(c.PIXEL_PACK_BUFFER,e),c.getBufferSubData(c.PIXEL_PACK_BUFFER,0,l),c.bindBuffer(c.PIXEL_PACK_BUFFER,null),l}function ag(n,e,t){const s=new Float32Array(e*t*4);return k(n,()=>n.readPixels(0,0,t,e,n.RGBA,n.FLOAT,s)),s}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ws{constructor(e){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];const t=y().getNumber("WEBGL_VERSION");if(e!=null?(this.gl=e,bp(t,e)):this.gl=_e(t),e=this.gl,y().getNumber("WEBGL_VERSION")===2){const o=e;this.createVertexArray=()=>k(o,()=>o.createVertexArray()),this.bindVertexArray=i=>k(o,()=>o.bindVertexArray(i)),this.deleteVertexArray=i=>k(o,()=>o.deleteVertexArray(i)),this.getVertexArray=()=>k(o,()=>o.getParameter(o.VERTEX_ARRAY_BINDING))}else if(e!=null){const o=e.getExtension("OES_vertex_array_object");if(o==null)throw new Error("All WebGL1 implementations are expected to offer OES_vertex_array_object.");this.createVertexArray=()=>k(e,()=>o.createVertexArrayOES()),this.bindVertexArray=i=>k(e,()=>o.bindVertexArrayOES(i)),this.deleteVertexArray=i=>k(e,()=>o.deleteVertexArrayOES(i)),this.getVertexArray=()=>k(e,()=>e.getParameter(o.VERTEX_ARRAY_BINDING_OES))}let s="WEBGL_color_buffer_float";const r="EXT_color_buffer_half_float";if(this.parallelCompilationExtension=this.gl.getExtension("KHR_parallel_shader_compile"),y().getNumber("WEBGL_VERSION")===1){const o="OES_texture_float",i="OES_texture_half_float";if(this.textureFloatExtension=kn(this.gl,o),Ie(this.gl,i))this.textureHalfFloatExtension=kn(this.gl,i);else if(y().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.");if(this.colorBufferFloatExtension=this.gl.getExtension(s),Ie(this.gl,r))this.colorBufferHalfFloatExtension=kn(this.gl,r);else if(y().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.")}else if(s="EXT_color_buffer_float",Ie(this.gl,s))this.colorBufferFloatExtension=this.gl.getExtension(s);else if(Ie(this.gl,r))this.colorBufferHalfFloatExtension=this.gl.getExtension(r);else throw new Error("GL context does not support color renderable floats");this.vertexBuffer=jm(this.gl),this.indexBuffer=qm(this.gl),this.framebuffer=Bp(this.gl),this.textureConfig=fr(this.gl,this.textureHalfFloatExtension)}get debug(){return y().getBool("DEBUG")}dispose(){if(this.disposed)return;this.program!=null&&console.warn("Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing."),this.outputTexture!=null&&console.warn("Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.");const e=this.gl;k(e,()=>e.finish()),k(e,()=>e.bindFramebuffer(e.FRAMEBUFFER,null)),k(e,()=>e.deleteFramebuffer(this.framebuffer)),k(e,()=>e.bindBuffer(e.ARRAY_BUFFER,null)),k(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,null)),k(e,()=>e.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(e,t){return this.throwIfDisposed(),Km(this.gl,e,t,this.textureConfig)}createFloat16MatrixTexture(e,t){return this.throwIfDisposed(),Ym(this.gl,e,t,this.textureConfig)}createUnsignedBytesMatrixTexture(e,t){return this.throwIfDisposed(),Qm(this.gl,e,t,this.textureConfig)}uploadPixelDataToTexture(e,t){this.throwIfDisposed(),ng(this.gl,e,t)}uploadDenseMatrixToTexture(e,t,s,r){this.throwIfDisposed(),tg(this.gl,e,t,s,r,this.textureConfig)}createFloat16PackedMatrixTexture(e,t){return this.throwIfDisposed(),Jm(this.gl,e,t,this.textureConfig)}createPackedMatrixTexture(e,t){return this.throwIfDisposed(),Zm(this.gl,e,t,this.textureConfig)}deleteMatrixTexture(e){this.throwIfDisposed(),this.outputTexture===e&&(Qr(this.gl,this.framebuffer),this.outputTexture=null),k(this.gl,()=>this.gl.deleteTexture(e))}downloadByteEncodedFloatMatrixFromOutputTexture(e,t,s){return this.downloadMatrixDriver(e,()=>og(this.gl,t,s,this.textureConfig))}downloadPackedMatrixFromBuffer(e,t,s,r,o,i){return ig(this.gl,e,t,s,r,o,i,this.textureConfig)}downloadFloat32MatrixFromBuffer(e,t){return rg(this.gl,e,t)}createBufferFromTexture(e,t,s){this.bindTextureToFrameBuffer(e);const r=sg(this.gl,t,s,this.textureConfig);return this.unbindTextureToFrameBuffer(),r}createAndWaitForFence(){const e=this.createFence(this.gl);return this.pollFence(e)}createFence(e){let t,s;if(y().getBool("WEBGL_FENCE_API_ENABLED")){const r=e,o=r.fenceSync(r.SYNC_GPU_COMMANDS_COMPLETE,0);e.flush(),s=()=>{const i=r.clientWaitSync(o,0,0);return i===r.ALREADY_SIGNALED||i===r.CONDITION_SATISFIED},t=o}else y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0?(t=this.beginQuery(),this.endQuery(),s=()=>this.isQueryAvailable(t,y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))):s=()=>!0;return{query:t,isFencePassed:s}}downloadMatrixFromPackedTexture(e,t,s){return this.downloadMatrixDriver(e,()=>ag(this.gl,t,s))}createProgram(e){this.throwIfDisposed();const t=this.gl;this.vertexShader==null&&(this.vertexShader=Xm(t));const s=Fp(t);k(t,()=>t.attachShader(s,this.vertexShader)),k(t,()=>t.attachShader(s,e)),Dp(t,s);const r=Object.assign(s,{vao:this.createVertexArray()});return this.debug&&gs(t,r),r}buildVao(e){this.setProgram(e),this.bindVertexArray(e.vao);const t=this.gl;k(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),eg(t,e,this.vertexBuffer)}deleteProgram(e){this.throwIfDisposed(),e===this.program&&(this.program=null),e!=null&&(k(this.gl,()=>this.gl.deleteProgram(e)),this.deleteVertexArray(e.vao))}setProgram(e){this.throwIfDisposed(),this.program=e,this.program!=null&&this.debug&&gs(this.gl,this.program),k(this.gl,()=>this.gl.useProgram(e))}getUniformLocation(e,t,s=!0){return this.throwIfDisposed(),s?Vp(this.gl,e,t):Up(this.gl,e,t)}getAttributeLocation(e,t){return this.throwIfDisposed(),k(this.gl,()=>this.gl.getAttribLocation(e,t))}getUniformLocationNoThrow(e,t){return this.throwIfDisposed(),this.gl.getUniformLocation(e,t)}setInputMatrixTexture(e,t,s){this.throwIfDisposed(),this.throwIfNoProgram(),Wp(this.gl,e,t,s)}setOutputMatrixTexture(e,t,s){this.setOutputMatrixTextureDriver(e,s,t)}setOutputPackedMatrixTexture(e,t,s){this.throwIfDisposed();const[r,o]=Yt(t,s);this.setOutputMatrixTextureDriver(e,r,o)}setOutputMatrixWriteRegion(e,t,s,r){this.setOutputMatrixWriteRegionDriver(s,e,r,t)}setOutputPackedMatrixWriteRegion(e,t,s,r){throw new Error("setOutputPackedMatrixWriteRegion not implemented.")}debugValidate(){this.program!=null&&gs(this.gl,this.program),An(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();const e=this.gl;if(this.debug){const t=this.getVertexArray();console.assert(t===this.program.vao,"VAO changed between setProgram and executeProgram!"),this.debugValidate()}k(e,()=>e.drawElements(e.TRIANGLES,6,e.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),k(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension==null&&(this.disjointQueryTimerExtension=kn(this.gl,y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2?"EXT_disjoint_timer_query_webgl2":"EXT_disjoint_timer_query")),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const s=this.gl,r=this.getQueryTimerExtensionWebGL2(),o=s.createQuery();return s.beginQuery(r.TIME_ELAPSED_EXT,o),o}const e=this.getQueryTimerExtensionWebGL1(),t=e.createQueryEXT();return e.beginQueryEXT(e.TIME_ELAPSED_EXT,t),t}endQuery(){if(y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const t=this.gl,s=this.getQueryTimerExtensionWebGL2();t.endQuery(s.TIME_ELAPSED_EXT);return}const e=this.getQueryTimerExtensionWebGL1();e.endQueryEXT(e.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(e){return await vr(()=>this.disposed||this.isQueryAvailable(e,y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))),this.getQueryTime(e,y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))}getQueryTime(e,t){if(t===0)return null;if(t===2){const s=this.gl;return s.getQueryParameter(e,s.QUERY_RESULT)/1e6}else{const s=this.getQueryTimerExtensionWebGL1();return s.getQueryObjectEXT(e,s.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(e,t){if(t===0)return!0;if(t===2){const s=this.gl,r=this.getQueryTimerExtensionWebGL2(),o=s.getQueryParameter(e,s.QUERY_RESULT_AVAILABLE);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(r.GPU_DISJOINT_EXT)),o&&!this.disjoint}else{const s=this.getQueryTimerExtensionWebGL1(),r=s.getQueryObjectEXT(e,s.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(s.GPU_DISJOINT_EXT)),r&&!this.disjoint}}pollFence(e){return new Promise(t=>{this.addItemToPoll(()=>e.isFencePassed(),()=>t())})}pollItems(){const e=cg(this.itemsToPoll.map(t=>t.isDoneFn));for(let t=0;t<=e;++t){const{resolveFn:s}=this.itemsToPoll[t];s()}this.itemsToPoll=this.itemsToPoll.slice(e+1)}addItemToPoll(e,t){if(this.itemsToPoll.push({isDoneFn:e,resolveFn:t}),this.itemsToPoll.length>1)return;let s;"setTimeoutCustom"in y().platform&&(s=y().platform.setTimeoutCustom.bind(y().platform)),vr(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,s)}bindTextureToFrameBuffer(e){this.throwIfDisposed(),xs(this.gl,e,this.framebuffer),this.debug&&An(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture!=null?(xs(this.gl,this.outputTexture,this.framebuffer),this.debug&&An(this.gl)):Qr(this.gl,this.framebuffer)}downloadMatrixDriver(e,t){this.bindTextureToFrameBuffer(e);const s=t();return this.unbindTextureToFrameBuffer(),s}setOutputMatrixTextureDriver(e,t,s){this.throwIfDisposed();const r=this.gl;xs(r,e,this.framebuffer),this.debug&&An(r),this.outputTexture=e,k(r,()=>r.viewport(0,0,t,s)),k(r,()=>r.scissor(0,0,t,s))}setOutputMatrixWriteRegionDriver(e,t,s,r){this.throwIfDisposed(),k(this.gl,()=>this.gl.scissor(e,t,s,r))}throwIfDisposed(){if(this.disposed)throw new Error("Attempted to use disposed GPGPUContext.")}throwIfNoProgram(){if(this.program==null)throw new Error("No GPU program is currently set.")}}function cg(n){let e=0;for(;e<n.length&&n[e]();++e);return e-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lg(n){const e=new Float32Array(n.length);for(let t=0;t<n.length;++t)e[t]=Math.abs(n[t]);return e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function he(n){return(e,t,s,r,o)=>{const i=ae(e,t),a=i.length,c=Z(i),l=E(i),u=pt(o,l),d=e.length,h=t.length,f=Z(e),p=Z(t),x=Xn(e,i),g=Xn(t,i);if(x.length+g.length===0)for(let m=0;m<u.length;++m)u[m]=n(s[m%s.length],r[m%r.length]);else for(let m=0;m<u.length;++m){const C=Xs(m,a,c),b=C.slice(-d);x.forEach(T=>b[T]=0);const w=Is(b,d,f),$=C.slice(-h);g.forEach(T=>$[T]=0);const N=Is($,h,p);u[m]=n(s[w],r[N])}return[u,i]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ug(n,e,t,s){if(s==="int32"){const r=Int32Array.from(n);return[e,"int32",r]}if(s==="bool"){const r=es([0],t),[o,i]=he((a,c)=>a!==c?1:0)(e,[],n,r,"bool");return[i,"bool",o]}throw new Error(`Error in Cast: failed to cast ${t} to ${s}`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dg=he((n,e)=>n+e);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hg(n,e,t,s,r){const o=E(s),i=nt(r,t);for(let a=0;a<n.length;a++){const c=n[a];if(c<0)throw new Error("Input x must be non-negative!");c>=r||(o>0?i[c]+=e[a]:i[c]+=1)}return i}function fg(n,e,t,s=!1){const r=n.shape[0],o=n.shape[1],i=ee([r,t],e.dtype);for(let a=0;a<r;a++)for(let c=0;c<o;c++){const l=n.get(a,c);if(l<0)throw new Error("Input x must be non-negative!");l>=t||(s?i.set(1,a,l):e.size>0?i.set(i.get(a,l)+e.get(a,c),a,l):i.set(i.get(a,l)+1,a,l))}return i}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pg=he((n,e)=>n&e);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ke(n){return(e,t,s)=>{const r=q(t,e.length);for(let o=0;o<e.length;++o)r[o]=n(e[o],s);return r}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mg=Ke(n=>Math.ceil(n));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gg(n,e,t,s){const r=q(t,E(e));if(s&&t!=="string"){let o=0;n.forEach(i=>{const a=E(i.shape);r.set(i.vals,o),o+=a})}else{let o=0;n.forEach(i=>{const a=t==="string"?Gt(i.vals):i.vals;let c=0;for(let l=0;l<i.shape[0];++l){const u=l*e[1]+o;for(let d=0;d<i.shape[1];++d)r[u+d]=a[c++]}o+=i.shape[1]})}return r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xg=he((n,e)=>n===e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cg=Ke(n=>Math.exp(n));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bg=Ke(n=>Math.expm1(n));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wg=Ke(n=>Math.floor(n));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yg=he((n,e)=>Math.floor(n/e));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $g(n,e,t,s,r,o,i,a,c){const l=ee([s,o],t);for(let u=0;u<s;u++){const d=[];let h=0;for(let f=0;f<r;f++){const p=n[u*r+f];h+=p*i[f],d.push(p)}if(h<0||h>=c/o)throw new Error(`Invalid indices: ${d} does not index into ${a}`);for(let f=0;f<o;f++)l.values[u*o+f]=e.get(...e.indexToLoc(h*o+f))}return l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vg(n,e,t){const s=ee(t,n.dtype);for(let r=0;r<s.size;++r){const i=s.indexToLoc(r).slice(),a=i[0],c=i[2],l=e.locToIndex([a,c]);i[2]=e.values[l];const u=n.locToIndex(i);0<=u&&u<n.values.length&&(s.values[r]=n.values[u])}return s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sg=he((n,e)=>n>e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ig=he((n,e)=>n>=e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rg=he((n,e)=>n<e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tg=he((n,e)=>n<=e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eg(n,e,t){const s=(e-n)/(t-1),r=nt(t,"float32");r[0]=n;for(let o=1;o<r.length;o++)r[o]=r[o-1]+s;return r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ng=Ke(n=>Math.log(n));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kg(n,e,t,s){const r=pt(s,E(t));for(let o=0;o<r.length;++o){const i=o*e;let a=n[i];for(let c=0;c<e;++c){const l=n[i+c];(Number.isNaN(l)||l>a)&&(a=l)}r[o]=a}return r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ag=he((n,e)=>Math.max(n,e));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fg=he((n,e)=>Math.min(n,e));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wa=he((n,e)=>n*e);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dg(n,e,t){const s=Xt(-1,t);return Wa([],e,s,n,t)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Og=he((n,e)=>n!==e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pg(n,e,t,s,r){const o=e.length,i=E(e),a=Z(e),c=Z(r),l=pt(t,E(r));for(let u=0;u<i;++u){const d=Xs(u,o,a),h=new Array(d.length);for(let p=0;p<h.length;p++)h[p]=d[s[p]];const f=Is(h,o,c);l[f]=n[u]}return l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _g(n,e,t,s){const[r,o]=He(n,s),i=ze(e,"int32"),a=nt(E(r),i),c=E(o);for(let l=0;l<a.length;++l){const u=l*c;let d=1;for(let h=0;h<c;++h)d*=t[u+h];a[l]=d}return{outVals:a,outShape:r,outDtype:i}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lg(n,e,t){n.forEach((s,r)=>{if(s<0||s>=t){const o=Xs(r,e.length,Z(e)).join(",");throw new Error(`indices[${o}] = ${s} is not in [0, ${t})`)}})}function Bg(n,e){for(let t=0;t<n.length;++t){const s=n[t],r=t===n.length-1?e:n[t+1].length;if(s.length===0)throw new Error("Ragged splits may not be empty");if(s[0]<0)throw new Error("Ragged splits must be non-negative");if(s[s.length-1]>r)throw new Error("Ragged splits must not point past values");for(let o=1;o<s.length;++o)if(s[o-1]>s[o])throw new Error("Ragged splits must be sorted in ascending order")}}function Mg(n,e,t,s){const r=[];let o=0;const i=e.length-1+t.length,a=new Array(i).fill(null).map(()=>[0]);Bg(t,s);let c=1;for(let l=0;l<e.length-1;++l){c*=e[l];const u=e[l+1];for(let d=1;d<c+1;++d)a[l].push(d*u)}for(let l=0;l<n.length;++l){let u=n[l],d=n[l]+1;for(let h=0;h<t.length;++h){const f=t[h],p=h+e.length-1;if(p>=0){const x=a[p],g=x[x.length-1]-f[u];for(let m=u;m<d;++m)a[p].push(f[m+1]+g)}u=f[u],d=f[d]}d!==u&&(r.push([u,d]),o+=d-u)}return{outSplits:a,valueSlices:r,numValues:o}}function Vg(n){const e=[];for(let t=0;t<n.length;++t){const s=n[t].length,r=q("int32",s);e.push(r),n[t].forEach((o,i)=>r[i]=o)}return e}function to(n,e){const t=n.slice(0,e);for(;t.length<e;)t.push(1);for(let s=e;s<n.length;s++)t[e-1]*=n[s];return t}function Ug(n,e,t,s,r,o){const i=to(e,2)[1],a=to(o,2)[1];let c=0;for(const l of t)for(let u=l[0];u<l[1];++u){for(let d=0;d<s;++d)r[c*a+d]=n[u*i+d];++c}}function Wg(n,e,t,s,r){const o=e.slice();o[0]=r;const i=q(t,E(o)),a=n.length,c=a===0?0:a/e[0];return Ug(n,e,s,c,i,o),[i,o]}function Gg(n,e,t,s,r,o,i,a){if(n.length===0)throw new Error("paramsNestedSplits must be non empty");if(e[0].length===0)throw new Error("Split tensors must not be scalars");const c=e[0][0]-1;if(Lg(o,i,c),s.length===0)throw new Error("params.rank must be nonzero");const l=s[0],{outSplits:u,valueSlices:d,numValues:h}=Mg(o,i,n,l),f=Vg(u),p=Wg(t,s,r,d,h);return[f,p[0],p[1]]}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const no=2147483647;function zg(n,e,t,s,r,o,i){if(e.length>1)throw new Error("starts must be a scalar or vector");if(r.length>1)throw new Error("limits must be a scalar or vector");if(i.length>1)throw new Error("deltas must be a scalar or vector");const a=e.length===0,c=r.length===0,l=i.length===0,u=[];a||u.push(e[0]),c||u.push(r[0]),l||u.push(i[0]);for(let g=1;g<u.length;++g)if(u[g]!==u[g-1])throw new Error("starts, limits, and deltas must have the same shape");const d=u.length===0?1:u[0],h=q("int32",d+1);h[0]=0;for(let g=0;g<d;++g){const m=a?n[0]:n[g],C=c?s[0]:s[g],b=l?o[0]:o[g];if(b===0)throw new Error("Requires delta != 0");let w;if(b>0&&C<m||b<0&&C>m)w=0;else if(w=Math.ceil(Math.abs((C-m)/b)),w>no)throw new Error(`Requires ((limit - start) / delta) <= ${no}`);h[g+1]=h[g]+w}const f=h[d],p=q(t,f);let x=0;for(let g=0;g<d;++g){const m=h[g+1]-h[g];let C=a?n[0]:n[g];const b=l?o[0]:o[g];for(let w=0;w<m;++w)p[x++]=C,C+=b}return[h,p]}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var ye=Oe;class qn{constructor(e,t,s,r,o,i,a,c,l,u){this.shape=e,this.shapeShape=t,this.values=s,this.valuesShape=r,this.valuesDType=o,this.defaultValue=i,this.defaultValueShape=a,this.rowPartitionValues=c,this.rowPartitionValuesShapes=l,this.rowPartitionTypes=Yi(u),this.raggedRank=Qi(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===ye.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===ye.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){const t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case ye.VALUE_ROWIDS:return qn.getMaxWidthValueRowID(t);case ye.ROW_SPLITS:return qn.getMaxWidthRowSplit(t);default:throw new Error(`Cannot handle partition type ${ye[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){const t=e.length;if(t===0||t===1)return 0;let s=0;for(let r=0;r<t-1;++r){const o=e[r+1]-e[r];o>s&&(s=o)}return s}static getMaxWidthValueRowID(e){const t=e.length;if(t===0)return 0;let s=0,r=e[0],o=0;for(let i=1;i<t;++i){const a=e[i];a!==r&&(r=a,o=Math.max(i-s,o),s=i)}return Math.max(t-s,o)}tensorShapeFromTensor(e,t,s=!0){if(t.length===0){if(e[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return ro(e,s)}calculateOutputSize(e){const t=this.valuesShape,s=this.defaultValueShape;Zi(s,t);const r=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=Ki(this.raggedRank,r,t);i[0]<0&&(i[0]=e);for(let a=1;a<=this.raggedRank;++a)i[a]<0&&(i[a]=this.getMaxWidth(a));return i}calculateFirstParentOutputIndex(e,t,s){const r=Math.min(e,s),o=[];let i=0;for(let a=0;a<r;++a,i+=t)o.push(i);for(let a=r;a<e;++a)o.push(-1);return I(o.length===e,()=>"Final length of result must be equal to firstDimension."),o}calculateOutputIndexRowSplit(e,t,s,r){const o=e.length,i=[];for(let a=0;a<o-1;++a){const c=e[a+1]-e[a];let l=Math.min(r,c),u=t[a];u===-1&&(l=0);for(let d=0;d<l;++d)i.push(u),u+=s;for(let d=0;d<c-l;++d)i.push(-1)}if(o>0&&i.length!==e[o-1])throw new Error("Invalid row split size.");return i}calculateOutputIndexValueRowID(e,t,s,r){const o=e.length,i=[];if(o===0)return[];let a=0,c=e[0];if(c>=t.length)throw new Error(`Got currentValueRowId=${c}, which is not less than ${t.length}`);let l=t[c];i.push(l);for(let u=1;u<o;++u){const d=e[u];if(d===c)l>=0&&(++a,a<r?l+=s:l=-1);else{if(a=0,c=d,d>=t.length)throw new Error(`Got nextValueRowId=${d} which is not less than ${t.length}`);l=t[d]}i.push(l)}if(i.length!==e.length)throw new Error("Invalid row ids.");return i}calculateOutputIndex(e,t,s,r){const o=this.getRowPartitionTensor(e),i=this.getRowPartitionTypeByDimension(e);switch(i){case ye.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(o,t,s,r);case ye.ROW_SPLITS:if(o.length-1>t.length)throw new Error(`Row partition size is greater than output size: ${o.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(o,t,s,r);default:throw new Error(`Unsupported partition type: ${ye[i]}`)}}getFirstDimensionSize(){const e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const t=this.rowPartitionTypes[0];switch(t){case ye.FIRST_DIM_SIZE:return e[0];case ye.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case ye.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${ye[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const t=this.getFirstDimensionSize(),s=this.calculateOutputSize(t),r=new Array(this.raggedRank+1);r[r.length-1]=1;for(let c=r.length-2;c>=0;--c)r[c]=r[c+1]*s[c+1];const o=ro(s,!1),i=q(this.valuesDType,E(o));if(r[0]*s[0]>0){let c=this.calculateFirstParentOutputIndex(t,r[0],s[0]);for(let l=1;l<=this.raggedRank;++l)c=this.calculateOutputIndex(l-1,c,r[l],s[l]);this.setOutput(this.raggedRank,c,i,o)}return[o,i]}setOutput(e,t,s,r){if(s.length===0)return;const o=this.values,i=s;let a=r.slice();a=a.slice(e+1);const c=E(a),l=t.length;let u=this.defaultValue;if(u.length!==c&&u.length!==1){const p=this.defaultValueShape;X(()=>{const x=nr(u,p);u=Xh(x,a).dataSync()})}let d=0,h=0,f=0;for(let p=0;p<=l;++p){let x=p<l?t[p]:-1;if(x===f){++f;continue}if(h<f){const g=o.subarray(d*c),m=i.subarray(h*c),C=(f-h)*c;so(m,g,C)}if(p>=l){const g=s.length;x=Math.floor(g/c)}if(x>f)if(this.defaultValue.length===1)i.subarray(f*c,x*c).fill(this.defaultValue[0]),f=x;else for(;x>f;){const g=i.slice(f*c);so(g,u,c),++f}x<0?(d=p+1,h=f):(d=p,h=f,f=h+1)}}}function so(n,e,t){for(let s=0;s<t;s++)n[s]=e[s]}function ro(n,e){const t=[];for(let s of n){if(s<0){if(!e)throw new Error(`Dimension ${s} must be >= 0`);if(s<-1)throw new Error(`Dimension ${s} must be >= -1`);s=-1}t.push(s)}return t}function Hg(n,e,t,s,r,o,i,a,c,l){return new qn(n,e,t,s,r,o,i,a,c,l).compute()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xg(n,e,t,s){const r=n===e,o=n<e&&t<0,i=e<n&&t>1;if(r||o||i)return nt(0,s);const a=Math.abs(Math.ceil((e-n)/t)),c=nt(a,s);e<n&&t===1&&(t=-1),c[0]=n;for(let l=1;l<c.length;l++)c[l]=c[l-1]+t;return c}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jg=Ke(n=>1/Math.sqrt(n));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qg(n,e,t,s,r,o,i,a,c,l){const u=[s/r,r],d=n.values,h=e.values;if(s===0)return ee(t,e.dtype);const f=c instanceof Gn?c:ee(u,e.dtype);typeof c=="string"||typeof c=="number"?f.values.fill(c):typeof c=="boolean"&&f.values.fill(+c);for(let p=0;p<o;p++){const x=[];let g=0;for(let m=0;m<i;m++){const C=d[p*i+m];x.push(C),g+=C*a[m]}if(g<0||g>=s/r)throw new Error(`Invalid indices: ${x} does not index into ${t}`);for(let m=0;m<r;m++)l?f.values[g*r+m]+=h[p*r+m]:f.values[g*r+m]=e.rank===0?h[0]:h[p*r+m]}return f}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kg=Ke(n=>1/(1+Math.exp(-n)));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yg(n,e,t,s,r){const o=ar(s,e,t),i=E(t),a=Z(s);if(o){const d=cr(e,a);return r==="string"?n.slice(d,d+i):n.subarray(d,d+i)}const c=r==="string"?Gt(n):n,l=ee(s,r,c),u=ee(t,r);for(let d=0;d<u.size;++d){const h=u.indexToLoc(d),f=h.map((p,x)=>p+e[x]);u.set(l.get(...f),...h)}return r==="string"?ka(u.values):u.values}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qg(n,e,t,s,r,o,i){const a=e[0],c=o[0],l=new Array(c),u=new Array(a),d=e[1];if(c===0){if(a!==0)throw new Error(ga(a));const g=q(t,0),m=q(r,0);return[g,[0,d],m,l,u]}let h=!0,f=0;const p=new Array(c).fill(0);for(let g=0;g<a;++g){const m=n[g*d];if(m<0)throw new Error(xa(g,m));if(m>=c)throw new Error(Ca(g,m,c));++p[m],h=h&&m>=f,f=m}let x=!0;for(let g=0;g<c;++g){const m=p[g]===0;l[g]=m,x=x&&!m,p[g]=Math.max(p[g],1),g>0&&(p[g]+=p[g-1])}if(x&&h){const g=n,m=s;for(let C=0;C<a;++C)u[C]=C;return[g,[a,d],m,l,u]}else{const g=p[c-1],m=q(t,g*d),C=q(r,g),b=new Array(c).fill(0);for(let w=0;w<a;++w){const $=n[w*d],N=b[$],T=($===0?0:p[$-1])+N;b[$]++;for(let v=0;v<d;++v)m[T*d+v]=n[w*d+v];C[T]=s[w],u[w]=T}for(let w=0;w<c;++w)if(b[w]===0){const N=w===0?0:p[w-1];m[N*d+0]=w;for(let T=1;T<d;++T)m[N*d+T]=0;C[N]=i}return[m,[g,d],C,l,u]}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zg(n,e,t,s,r){const o=E(s),i=e[0],a=r.length,c=[];let l=1,u=-1;for(let g=0;g<a;++g){const m=r[g];if(m===-1){if(u!==-1)throw new Error(ba(u,g));u=g,c.push(1)}else{if(m<0)throw new Error(wa(g,m));l*=m,c.push(m)}}if(u!==-1){if(l<=0)throw new Error(ya());const g=Math.trunc(o/l);if(l*g!==o)throw new Error($a(s,c));c[u]=g}if(E(c)!==o)throw new Error(va(s,c));const h=s.length,f=[];if(h>0){f[h-1]=1;for(let g=h-2;g>=0;--g)f[g]=f[g+1]*s[g+1]}const p=[];if(a>0){p[a-1]=1;for(let g=a-2;g>=0;--g)p[g]=p[g+1]*c[g+1]}const x=q(t,i*a);for(let g=0;g<i;++g){let m=0;for(let C=0;C<h;++C)m+=n[g*h+C]*f[C];for(let C=0;C<a;++C)x[g*a+C]=Math.trunc(m/p[C]),m%=p[C]}return[x,[i,a],c]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jg(n,e,t,s,r,o=!1,i=0){const a=s.length,c=[e[0],n.length/e[0]],l=c[1],d=a>0?r[a-1]+1:0;if(d<0)throw new Error(Vs());const h=e.slice();h[0]=d;const f=h.reduce((b,w)=>b*w,1),p=q(t,f);if(a===0)return d>0&&p.fill(i),[p,h];if(d<=0)throw new Error(Vs());let x=0,g=1,m=0,C=r[x];for(;;){let b=0;if(g<a){if(b=r[g],C===b){++g;continue}if(C>=b)throw new Error(Sa())}if(C<0||C>=d)throw new Error(Ia(C,d));C>m&&p.fill(i,m*l,C*l);for(let w=x;w<g;++w){const $=s[w];if($<0||$>=c[0])throw new Error(Ra(w,s[w],c[0]));for(let N=0;N<l;N++)p[C*l+N]+=n[$*l+N]}if(o)for(let w=0;w<l;w++)p[C*l+w]/=g-x;if(x=g,++g,m=C+1,C=b,g>a)break}return m<d&&p.fill(i,m*l,d*l),[p,h]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ex=Ke(n=>Math.sqrt(n));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tx=he((n,e)=>{const t=n-e;return t*t});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nx=Ke((n,e)=>{const{pattern:t,replaceGlobal:s,rewrite:r}=e;return n.replace(new RegExp(t,s?"g":""),r)});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sx(n,e,t,s){const r=ee(n,e.dtype);for(let o=0;o<r.size;o++){const i=r.indexToLoc(o),a=new Array(i.length);for(let c=0;c<a.length;c++)a[c]=i[c]*t[c]+s[c];r.set(e.get(...a),...i)}return r}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class rx{constructor(e,t,s,r,o,i){this.separator=ht(e),this.nGramWidths=t,this.leftPad=ht(s),this.rightPad=ht(r),this.padWidth=o,this.preserveShort=i}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){const s=this.getPadWidth(t);return Math.max(0,e+2*s-t+1)}createNGrams(e,t,s,r,o,i){for(let a=0;a<o;++a){const c=this.getPadWidth(i),l=Math.max(0,c-a),u=Math.max(0,c-(o-(a+1))),d=i-(l+u),h=t+(l>0?0:a-c);let f=0;f+=l*this.leftPad.length;for(let C=0;C<d;++C)f+=e[h+C].length;f+=u*this.rightPad.length;const p=l+u+d-1;f+=p*this.separator.length,s[r+a]=new Uint8Array(f);const x=s[r+a];let g=0;const m=C=>C.forEach(b=>x[g++]=b);for(let C=0;C<l;++C)m(this.leftPad),m(this.separator);for(let C=0;C<d-1;++C)m(e[h+C]),m(this.separator);if(d>0){m(e[h+d-1]);for(let C=0;C<u;++C)m(this.separator),m(this.rightPad)}else{for(let C=0;C<u-1;++C)m(this.rightPad),m(this.separator);m(this.rightPad)}}}compute(e,t){const s=e.length,r=t.length;if(r>0){let c=t[0];if(c!==0)throw new Error(`First split value must be 0, got ${c}`);for(let l=1;l<r;++l){let u=t[l]>=c;if(u=u&&t[l]<=s,!u)throw new Error(`Invalid split value ${t[l]}, must be in [${c}, ${s}]`);c=t[l]}if(c!==s)throw new Error(`Last split value must be data size. Expected ${s}, got ${c}`)}const o=r-1,i=q("int32",r);if(s===0||r===0){const c=new Array(s);for(let l=0;l<=o;++l)i[l]=0;return[c,i]}i[0]=0;for(let c=1;c<=o;++c){const l=t[c]-t[c-1];let u=0;this.nGramWidths.forEach(d=>{u+=this.getNumNGrams(l,d)}),this.preserveShort&&l>0&&u===0&&(u=1),i[c]=i[c-1]+u}const a=new Array(i[o]);for(let c=0;c<o;++c){const l=t[c];let u=i[c];if(this.nGramWidths.forEach(d=>{const h=t[c+1]-t[c],f=this.getNumNGrams(h,d);this.createNGrams(e,l,a,u,f,d),u+=f}),this.preserveShort&&u===i[c]){const d=t[c+1]-t[c];if(d===0)continue;const h=d+2*this.padWidth;this.createNGrams(e,l,a,u,1,h)}}return[a,i]}}function ox(n,e,t,s,r,o,i,a){return new rx(t,s,r,o,i,a).compute(n,e)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ix(n,e,t,s){if(!n.length)return;if(e.length===0){for(let o=0;o<n.length;++o)s.push(n.subarray(o,o+1));return}if(e.length===1){const o=e[0];let i=n.indexOf(o);for(;i!==-1;){const a=n.subarray(0,i);(!t||a.length!==0)&&s.push(a),n=n.subarray(i+1),i=n.indexOf(o)}(!t||n.length!==0)&&s.push(n);return}let r=0;for(let o=0;o<n.length+1;o++)if(o===n.length||e.indexOf(n[o])!==-1){const i=n.subarray(r,o);(!t||i.length!==0)&&s.push(i),r=o+1}}function ax(n,e,t){const s=n.length,r=[];let o=0,i=0;const a=new Array(s);for(let h=0;h<s;++h){const f=r.length;ix(n[h],e,t,r);const p=r.length-f;a[h]=p,o+=p,i=Math.max(i,p)}const c=q("int32",o*2),l=new Array(o),u=[s,i];let d=0;for(let h=0;h<s;++h)for(let f=0;f<a[h];++f)c[d*2]=h,c[d*2+1]=f,l[d]=r[d],++d;return[c,l,u]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cx(n,e){const t=q("int32",n.length);for(let s=0;s<n.length;++s)t[s]=Ld(n[s]).modulo(e).getLowBitsUnsigned();return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lx=he((n,e)=>n-e);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ux(n,e){const t=new Array(n.rank);for(let r=0;r<t.length;r++)t[r]=n.shape[r]*e[r];const s=ee(t,n.dtype);for(let r=0;r<s.values.length;++r){const o=s.indexToLoc(r),i=new Array(n.rank);for(let c=0;c<i.length;c++)i[c]=o[c]%n.shape[c];const a=n.locToIndex(i);s.values[r]=n.values[a]}return s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ln=(n,e)=>{const t=e.value-n.value;return t===0?n.index-e.index:t};function Ga(n,e,t=0,s=n.length-1){for(;s>t;){if(s-t>600){const a=s-t+1,c=e-t+1,l=Math.log(a),u=.5*Math.exp(2*l/3),d=.5*Math.sqrt(l*u*(a-u)/a)*Math.sign(c-a/2),h=Math.max(t,Math.floor(e-c*u/a+d)),f=Math.min(s,Math.floor(e+(a-c)*u/a+d));Ga(n,e,h,f)}const r=n[e];let o=t,i=s;for(rn(n,t,e),ln(n[s],r)>0&&rn(n,t,s);o<i;){for(rn(n,o,i),o++,i--;ln(n[o],r)<0;)o=o+1;for(;ln(n[i],r)>0;)i=i-1}ln(n[t],r)===0?rn(n,t,i):(i=i+1,rn(n,i,s)),i<=e&&(t=i+1),e<=i&&(s=i-1)}}function dx(n,e,t,s,r){const o=e[e.length-1],[i,a]=[n.length/o,o],c=pt(t,i*s),l=pt("int32",i*s);for(let d=0;d<i;d++){const h=d*a,f=n.subarray(h,h+a);let p=new Array(f.length);f.forEach((C,b)=>p[b]={value:C,index:b}),s<p.length&&(Ga(p,s),p=p.slice(0,s)),r&&p.sort(ln);const x=d*s,g=c.subarray(x,x+s),m=l.subarray(x,x+s);for(let C=0;C<s;C++)g[C]=p[C].value,m[C]=p[C].index}const u=e.slice();return u[u.length-1]=s,[ee(u,t,c),ee(u,"int32",l)]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hx(n,e,t,s){const r=de(e,t)[0],o=[1,t[0],1];for(let p=0;p<r;p++)o[0]*=t[p];o[1]=t[r];for(let p=r+1;p<t.length;p++)o[2]*=t[p];const i=new Map,a=new Int32Array(t[r]),c=new Gn(o,s,n),l=[],u=o[0]===1&&o[2]===1;for(let p=0;p<t[r];p++){let x;if(u)x=n[p].toString();else{const m=[];for(let C=0;C<o[0];C++)for(let b=0;b<o[2];b++)m.push(c.get(C,p,b));x=m.join(",")}const g=i.get(x);if(g!=null)a[p]=g;else{const m=i.size;i.set(x,m),a[p]=m,l.push(p)}}const d=o.slice();d[1]=i.size;const h=new Gn(d,s);l.forEach((p,x)=>{for(let g=0;g<o[0];g++)for(let m=0;m<o[2];m++)h.set(c.get(g,p,m),g,x,m)});const f=t.slice();return f[r]=d[1],{outputValues:h.values,outputShape:f,indices:a}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fx=Object.freeze(Object.defineProperty({__proto__:null,addImpl:dg,bincountImpl:hg,bincountReduceImpl:fg,bitwiseAndImpl:pg,castImpl:ug,ceilImpl:mg,concatImpl:gg,equalImpl:xg,expImpl:Cg,expm1Impl:bg,floorDivImpl:yg,floorImpl:wg,gatherNdImpl:$g,gatherV2Impl:vg,greaterEqualImpl:Ig,greaterImpl:Sg,lessEqualImpl:Tg,lessImpl:Rg,linSpaceImpl:Eg,logImpl:Ng,maxImpl:kg,maximumImpl:Ag,minimumImpl:Fg,multiplyImpl:Wa,negImpl:Dg,notEqualImpl:Og,prodImpl:_g,raggedGatherImpl:Gg,raggedRangeImpl:zg,raggedTensorToTensorImpl:Hg,rangeImpl:Xg,rsqrtImpl:jg,scatterImpl:qg,sigmoidImpl:Kg,simpleAbsImpl:lg,sliceImpl:Yg,sparseFillEmptyRowsImpl:Qg,sparseReshapeImpl:Zg,sparseSegmentReductionImpl:Jg,sqrtImpl:ex,squaredDifferenceImpl:tx,staticRegexReplaceImpl:nx,stridedSliceImpl:sx,stringNGramsImpl:ox,stringSplitImpl:ax,stringToHashBucketFastImpl:cx,subImpl:lx,tileImpl:ux,topKImpl:dx,transposeImpl:Pg,uniqueImpl:hx},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const{addImpl:px,bincountImpl:za,bincountReduceImpl:mx,bitwiseAndImpl:gx,castImpl:xx,ceilImpl:Cx,concatImpl:bx,equalImpl:wx,expImpl:yx,expm1Impl:$x,floorImpl:vx,gatherNdImpl:Sx,gatherV2Impl:Ix,greaterImpl:Rx,greaterEqualImpl:Tx,lessImpl:Ex,lessEqualImpl:Nx,linSpaceImpl:kx,logImpl:Ax,maxImpl:Fx,maximumImpl:Dx,minimumImpl:Ox,multiplyImpl:Px,negImpl:_x,notEqualImpl:Lx,prodImpl:Bx,raggedGatherImpl:Mx,raggedRangeImpl:Vx,raggedTensorToTensorImpl:Ux,rangeImpl:Wx,rsqrtImpl:Gx,scatterImpl:zx,sigmoidImpl:Hx,simpleAbsImpl:Ha,sliceImpl:Xx,sparseFillEmptyRowsImpl:jx,sparseReshapeImpl:qx,sparseSegmentReductionImpl:Xa,sqrtImpl:Kx,staticRegexReplaceImpl:Yx,stridedSliceImpl:Qx,stringNGramsImpl:Zx,stringSplitImpl:Jx,stringToHashBucketFastImpl:e0,subImpl:t0,tileImpl:n0,topKImpl:s0,transposeImpl:xr,uniqueImpl:r0}=fx;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ja(n,e){return["x","y","z","w","u","v"].slice(0,e).map(t=>`${n}.${t}`)}function ie(n,e){return e===1?[n]:ja(n,e)}function o0(n,e){if(n===1)return"rc";let t="";for(let s=0;s<n;s++)t+=e[s],s<n-1&&(t+=",");return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class i0{constructor(e){if(this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.enableShapeUniforms=se(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{const t=ie("rc",this.rank),s=U(this.rank),r=this.getOutOfBoundsCondition(t),o=this.getSetup(t),i=this.getOutput(t);this.userCode=`
        void main() {
          ${s} rc = getOutputCoords();

          if(${r}) {
            setOutput(vec4(0));
          } else {
            ${o}

            setOutput(vec4(${i}));
          }
        }
      `}}getSourceCoordsArr(e){const t=[];for(let s=0;s<=1;s++)for(let r=0;r<=1;r++){let o=`${s===0?"r":"rp1"}, ${r===0?"c":"cp1"}`;for(let i=2;i<this.rank;i++)o=`${e[e.length-1-i]},`+o;t.push(o)}return t}getOutOfBoundsCondition(e){if(this.rank===1)return`rc > ${this.enableShapeUniforms?"outShape":this.outputShape[0]}`;let t="";for(let s=this.rank-2;s<this.rank;s++)t+=`${e[s]} >= ${this.enableShapeUniforms?`outShape[${s}]`:this.outputShape[s]}`,s<this.rank-1&&(t+="||");return t}getSetup(e){if(this.rank===1)return"";const t=e.slice(-2),s=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],r=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${t[0]};
      int c = ${t[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${s};
      bool rEdge = rp1 >= ${r};
    `}getOutput(e){const t=this.getSourceCoordsArr(e);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?"outShape":this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${t[0]}),
            cEdge ? 0. : getA(${t[1]}),
            rEdge ? 0. : getA(${t[2]}),
            rEdge || cEdge ? 0. : getA(${t[3]})`}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class qa{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec3"}],this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length);let s="";for(let r=0;r<4;r++){let o="thisRC = rc;";r%2===1&&(o+="thisRC.z += 1;"),r>1&&(o+="thisRC.y += 1;"),s+=`
        ${o}
        ${r>0?"if(thisRC.y < rows && thisRC.z < cols){":""}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${r}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${r>0?"}":""}
      `}this.userCode=`
      ${a0(t,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?mr():pr(e)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?"outShape[1]":e[1]};
        int cols = ${this.enableShapeUniforms?"outShape[2]":e[2]};

        ${s}

        setOutput(result);
      }
    `}}function a0(n,e){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${e?em(["r","c","d"],"inputShape"):Rt(["r","c","d"],n)}
      return ivec3(r, c, d);
    }
  `}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class c0{constructor(e){this.gpgpu=e,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(e,t,s){const r=io(t,s),o=ao(e,r,s);o in this.freeTextures||(this.freeTextures[o]=[]),o in this.usedTextures||(this.usedTextures[o]=[]);const i=oo(e,r,this.gpgpu.gl,this.gpgpu.textureConfig,s);if(this.freeTextures[o].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=i,this.log();const c=this.freeTextures[o].pop();return this.usedTextures[o].push(c),c}let a;return r===Q.PACKED_2X2_FLOAT32?a=this.gpgpu.createPackedMatrixTexture(e[0],e[1]):r===Q.PACKED_2X2_FLOAT16?a=this.gpgpu.createFloat16PackedMatrixTexture(e[0],e[1]):r===Q.UNPACKED_FLOAT32?a=this.gpgpu.createFloat32MatrixTexture(e[0],e[1]):r===Q.UNPACKED_FLOAT16?a=this.gpgpu.createFloat16MatrixTexture(e[0],e[1]):r===Q.PACKED_4X1_UNSIGNED_BYTE&&(a=this.gpgpu.createUnsignedBytesMatrixTexture(e[0],e[1])),this.usedTextures[o].push(a),this.numUsedTextures++,this._numBytesAllocated+=i,this.log(),a}releaseTexture(e,t,s,r){if(this.freeTextures==null)return;const o=io(s,r),i=ao(t,o,r);i in this.freeTextures||(this.freeTextures[i]=[]);const a=oo(t,o,this.gpgpu.gl,this.gpgpu.textureConfig,r),c=y().getNumber("WEBGL_DELETE_TEXTURE_THRESHOLD");c!==-1&&this._numBytesAllocated>c?(this.gpgpu.deleteMatrixTexture(e.texture),this._numBytesAllocated-=a):(this.freeTextures[i].push(e),this.numFreeTextures++,this._numBytesFree+=a),this.numUsedTextures--;const l=this.usedTextures[i],u=l&&l.indexOf(e);if(u==null||u<0)throw new Error("Cannot release a texture that was never provided by this texture manager");l[u]=l[l.length-1],l.pop(),this.log()}log(){if(!this.logEnabled)return;const e=this.numFreeTextures+this.numUsedTextures;console.log("Free/Used",`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${e})`);const t=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*t)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(const e in this.freeTextures)this.freeTextures[e].forEach(t=>{this.gpgpu.deleteMatrixTexture(t.texture)});for(const e in this.usedTextures)this.usedTextures[e].forEach(t=>{this.gpgpu.deleteMatrixTexture(t.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}}function l0(n,e){const t=n;if(e===t.R32F)return 4;if(e===t.R16F)return 2;if(e===t.RGBA32F)return 16;if(e===n.RGBA)return 16;if(e===t.RGBA16F)return 8;if(e===t.RGBA8)return 4;throw new Error(`Unknown internal format ${e}`)}function oo(n,e,t,s,r){const o=u0(e,s);let i;if(r){const[c,l]=Yt(n[0],n[1]);i=c*l}else{const[c,l]=yn(n[0],n[1]);i=c*l}const a=l0(t,o);return i*a}function u0(n,e){switch(n){case Q.PACKED_2X2_FLOAT32:return Va(e);case Q.PACKED_2X2_FLOAT16:return Ua(e);case Q.UNPACKED_FLOAT32:return La(e);case Q.UNPACKED_FLOAT16:return Ba(e);case Q.PACKED_4X1_UNSIGNED_BYTE:return Ma(e);default:throw new Error(`Unknown physical texture type ${n}`)}}function d0(n){return y().getBool("WEBGL_RENDER_FLOAT32_ENABLED")?n?Q.PACKED_2X2_FLOAT32:Q.UNPACKED_FLOAT32:n?Q.PACKED_2X2_FLOAT16:Q.UNPACKED_FLOAT16}function io(n,e){if(n===Ce.UPLOAD)return Q.PACKED_2X2_FLOAT32;if(n===Ce.RENDER||n==null)return d0(e);if(n===Ce.DOWNLOAD||n===Ce.PIXELS)return Q.PACKED_4X1_UNSIGNED_BYTE;throw new Error(`Unknown logical texture type ${n}`)}function ao(n,e,t){return`${n[0]}_${n[1]}_${e}_${t}`}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ue{constructor(e,t){this.variableNames=["A"],this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${t}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}}const Ne="if (isnan(x)) return x;",h0="return x;",co="return abs(x);",f0="return (x >= 0.0) ? x : (exp(x) - 1.0);",p0=Ne+`
  return (x < 0.0) ? 0.0 : x;
`,m0=Ne+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,Ze="return x;",g0="return 1.0 / (1.0 + exp(-1.0 * x));";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const x0="return x;",C0=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,b0=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,w0=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,y0="return 1.0 / (1.0 + exp(-1.0 * x));";class et{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${t}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $0{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length);const t=e.length,s=ie("rc",t),r=U(t),o=o0(t,s),i=s.slice(-2),a=t<=1?"rc":`vec2(${i.join(",")})`;this.userCode=`
      void main() {
        ${r} rc = getOutputCoords();
        vec4 packedInput = getA(${o});

        setOutput(getChannel(packedInput, ${a}));
      }
    `}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const v0=Cf,S0=1e-7,I0=1e-4,On={};function R0(n){return n in On||(On[n]={}),On[n]}const T0=y().getNumber("CPU_HANDOFF_SIZE_THRESHOLD"),E0=600;function N0(){return y().global.screen==null?1024:y().global.screen.height*y().global.screen.width*window.devicePixelRatio*E0/1024/1024}class os extends Eo{nextDataId(){return os.nextDataId++}constructor(e){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!y().getBool("HAS_WEBGL"))throw new Error("WebGL is not supported on this device");let t;if(e!=null){if(e instanceof ws)t=e;else{const s=_e(y().getNumber("WEBGL_VERSION"),e);t=new ws(s)}this.binaryCache={},this.gpgpuCreatedLocally=!1}else{const s=_e(y().getNumber("WEBGL_VERSION"));t=new ws(s),this.binaryCache=R0(y().getNumber("WEBGL_VERSION")),this.gpgpuCreatedLocally=!0}this.gpgpu=t,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new c0(this.gpgpu),this.numMBBeforeWarning=N0(),this.texData=new Nc(this,Qe())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(e,t,s,r,o,i){const a=this.makeTensorInfo(t,s),c=this.texData.get(a.dataId);c.isPacked=!1,c.texture={texture:e,texShape:[r,o]},c.texShape=[r,o];const l=Fn(t),u=new eo(l,!1,i),d=this.runWebGLProgram(u,[a],s,[[r,o]]);return d.shape=t,c.texture=null,this.disposeIntermediateTensorInfo(a),d.dataId}write(e,t,s){if((y().getBool("WEBGL_CHECK_NUMERICAL_PROBLEMS")||y().getBool("DEBUG"))&&this.checkNumericalProblems(e),s==="complex64"&&e!=null)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const r={id:this.nextDataId()};return this.texData.set(r,{shape:t,dtype:s,values:e,usage:Ce.UPLOAD,refCount:1}),r}refCount(e){return this.texData.has(e)?this.texData.get(e).refCount:0}incRef(e){const t=this.texData.get(e);t.refCount++}decRef(e){if(this.texData.has(e)){const t=this.texData.get(e);t.refCount--}}move(e,t,s,r,o){if(y().getBool("DEBUG")&&this.checkNumericalProblems(t),r==="complex64")throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.texData.set(e,{shape:s,dtype:r,values:t,usage:Ce.UPLOAD,refCount:o})}disposeIntermediateTensorInfo(e){this.disposeData(e.dataId)}readSync(e){const t=this.texData.get(e),{values:s,dtype:r,complexTensorInfos:o,slice:i,shape:a,isPacked:c}=t;if(i!=null){let h;c?h=new et(a,Ze):h=new Ue(a,Ze);const f=this.runWebGLProgram(h,[{dataId:e,shape:a,dtype:r}],r),p=this.readSync(f.dataId);return this.disposeIntermediateTensorInfo(f),p}if(s!=null)return this.convertAndCacheOnCPU(e);if(r==="string")return s;const l=this.activeTimers!=null;let u;l&&(u=Ae());let d;if(r==="complex64"){const h=this.readSync(o.real.dataId),f=this.readSync(o.imag.dataId);d=Ms(h,f)}else d=this.getValuesFromTexture(e);return l&&(this.downloadWaitMs+=Ae()-u),this.convertAndCacheOnCPU(e,d)}async read(e){if(this.pendingRead.has(e)){const p=this.pendingRead.get(e);return new Promise(x=>p.push(x))}const t=this.texData.get(e),{values:s,shape:r,slice:o,dtype:i,complexTensorInfos:a,isPacked:c}=t;if(o!=null){let p;c?p=new et(r,Ze):p=new Ue(r,Ze);const x=this.runWebGLProgram(p,[{dataId:e,shape:r,dtype:i}],i),g=this.read(x.dataId);return this.disposeIntermediateTensorInfo(x),g}if(s!=null)return this.convertAndCacheOnCPU(e);if(y().getBool("DEBUG")&&!y().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")&&y().getNumber("WEBGL_VERSION")===2)throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.");let l=null,u;if(i!=="complex64"&&y().get("WEBGL_BUFFER_SUPPORTED")){u=this.decode(e);const p=this.texData.get(u.dataId);l=this.gpgpu.createBufferFromTexture(p.texture.texture,...Nn(r))}this.pendingRead.set(e,[]),i!=="complex64"&&await this.gpgpu.createAndWaitForFence();let d;if(i==="complex64"){const p=await Promise.all([this.read(a.real.dataId),this.read(a.imag.dataId)]),x=p[0],g=p[1];d=Ms(x,g)}else if(l==null)d=this.getValuesFromTexture(e);else{const p=E(r);d=this.gpgpu.downloadFloat32MatrixFromBuffer(l,p)}if(u!=null&&this.disposeIntermediateTensorInfo(u),l!=null){const p=this.gpgpu.gl;k(p,()=>p.deleteBuffer(l))}const h=this.convertAndCacheOnCPU(e,d),f=this.pendingRead.get(e);return this.pendingRead.delete(e),f.forEach(p=>p(h)),this.pendingDisposal.has(e)&&(this.pendingDisposal.delete(e),this.disposeData(e)&&Qe().removeDataId(e,this),this.pendingDeletes--),h}readToGPU(e,t={}){const s=this.texData.get(e),{values:r,shape:o,slice:i,dtype:a,isPacked:c,texture:l}=s;if(a==="complex64")throw new Error("Does not support reading texture for complex64 dtype.");if(i!=null){let f;c?f=new et(o,Ze):f=new Ue(o,Ze);const p=this.runWebGLProgram(f,[{dataId:e,shape:o,dtype:a}],a),x=this.readToGPU(p,t);return this.disposeIntermediateTensorInfo(p),x}if(l==null)throw r!=null?new Error("Data is not on GPU but on CPU."):new Error("There is no data on GPU or CPU.");const u=this.decode(e,t.customTexShape),d=Qe().makeTensorFromTensorInfo(u),h=this.texData.get(u.dataId);return Object.assign({tensorRef:d},h.texture)}bufferSync(e){const t=this.readSync(e.dataId);if(e.dtype==="string")try{const s=t.map(r=>Vt(r));return ee(e.shape,e.dtype,s)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return ee(e.shape,e.dtype,t)}checkNumericalProblems(e){if(e!=null)for(let t=0;t<e.length;t++){const s=e[t];if(!Tp(s))throw y().getBool("WEBGL_RENDER_FLOAT32_CAPABLE")?Error(`The value ${s} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${s} cannot be represented on this device.`)}}getValuesFromTexture(e){const{shape:t,dtype:s,isPacked:r}=this.texData.get(e),o=E(t);if(y().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")){const h=this.decode(e),f=this.texData.get(h.dataId),p=this.gpgpu.downloadMatrixFromPackedTexture(f.texture.texture,...Nn(t)).subarray(0,o);return this.disposeIntermediateTensorInfo(h),p}const i=y().getBool("WEBGL_PACK")&&r===!0,a=i?Fn(t):t,c=i?new Gm(a):new Wm(a),l=this.runWebGLProgram(c,[{shape:a,dtype:s,dataId:e}],"float32"),u=this.texData.get(l.dataId),d=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(u.texture.texture,u.texShape[0],u.texShape[1]).subarray(0,o);return this.disposeIntermediateTensorInfo(l),d}timerAvailable(){return y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0}time(e){const t=this.activeTimers,s=[];let r=!1;this.programTimersStack==null?(this.programTimersStack=s,r=!0):this.activeTimers.push(s),this.activeTimers=s,e();const o=mt(this.activeTimers.map(c=>c.query)).filter(c=>c!=null),i=mt(this.activeTimers.map(c=>c.name)).filter(c=>c!=null);this.activeTimers=t,r&&(this.programTimersStack=null);const a={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if(y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0){const c=await Promise.all(o);a.kernelMs=kc(c),a.getExtraProfileInfo=()=>c.map((l,u)=>({name:i[u],ms:l})).map(l=>`${l.name}: ${l.ms}`).join(", ")}else a.kernelMs={error:"WebGL query timers are not supported in this environment."};return this.uploadWaitMs=0,this.downloadWaitMs=0,a})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.beginQuery():{startMs:Ae(),endMs:null}}endTimer(e){return y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?(this.gpgpu.endQuery(),e):(e.endMs=Ae(),e)}async getQueryTime(e){if(y().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0)return this.gpgpu.waitForQueryAndGetTime(e);const t=e;return t.endMs-t.startMs}disposeData(e,t=!1){if(this.pendingDisposal.has(e))return!1;if(!this.texData.has(e))return!0;if(t?this.texData.get(e).refCount=0:this.texData.get(e).refCount--,!t&&this.texData.get(e).refCount>0)return!1;if(this.pendingRead.has(e))return this.pendingDisposal.add(e),this.pendingDeletes++,!1;this.releaseGPUData(e);const{complexTensorInfos:s}=this.texData.get(e);return s!=null&&(this.disposeData(s.real.dataId,t),this.disposeData(s.imag.dataId,t)),this.texData.delete(e),!0}releaseGPUData(e){const{texture:t,dtype:s,texShape:r,usage:o,isPacked:i,slice:a}=this.texData.get(e),c=a&&a.origDataId||e,l=this.dataRefCount.get(c);l>1?this.dataRefCount.set(c,l-1):(this.dataRefCount.delete(c),t!=null&&(this.numBytesInGPU-=this.computeBytes(r,s),this.textureManager.releaseTexture(t,r,o,i)));const u=this.texData.get(e);u.texture=null,u.texShape=null,u.isPacked=!1,u.slice=null}getTexture(e){return this.uploadToGPU(e),this.texData.get(e).texture.texture}getDataInfo(e){return this.texData.get(e)}shouldExecuteOnCPU(e,t=T0){return y().getBool("WEBGL_CPU_FORWARD")&&e.every(s=>this.texData.get(s.dataId).texture==null&&E(s.shape)<t)}getGPGPUContext(){return this.gpgpu}where(e){Pe("tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead");const t=e.dataSync();return v0(e.shape,t)}packedUnaryOp(e,t,s){const r=new et(e.shape,t),o=this.compileAndRun(r,[e],s);return Qe().makeTensorFromTensorInfo(o)}abs(e){if(this.shouldExecuteOnCPU([e])&&e.dtype!=="complex64"){const r=Ha(this.texData.get(e.dataId).values);return this.makeOutput(e.shape,e.dtype,r)}if(y().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(e,co,e.dtype);const t=new Ue(e.shape,co),s=this.compileAndRun(t,[e]);return Qe().makeTensorFromTensorInfo(s)}makeTensorInfo(e,t,s){let r;if(t==="string"&&s!=null&&s.length>0&&Zn(s[0])){const o=s.map(i=>ht(i));r=this.write(o,e,t)}else r=this.write(s,e,t);return this.texData.get(r).usage=null,{dataId:r,shape:e,dtype:t}}makeOutput(e,t,s){return Qe().makeTensorFromTensorInfo(this.makeTensorInfo(e,t,s),this)}unpackTensor(e){const t=new $0(e.shape);return this.runWebGLProgram(t,[e],e.dtype)}packTensor(e){const t=new i0(e.shape);return this.runWebGLProgram(t,[e],e.dtype,null,!0)}packedReshape(e,t){const s=[zt(e.shape),...Ht(e.shape)],r={dtype:e.dtype,shape:s,dataId:e.dataId},o=[zt(t),...Ht(t)],i=new qa(o,s),a=!0,c=[s],l=this.runWebGLProgram(i,[r],e.dtype,c,a);return{dataId:l.dataId,shape:t,dtype:l.dtype}}decode(e,t){const s=this.texData.get(e),{isPacked:r,shape:o,dtype:i}=s;if(t!=null){const h=E(o),f=t[0]*t[1]*4;I(h<=f,()=>"customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.")}const a=Fn(o);let c;r?c=new Um(a):c=new Vm(a);const l=!0,u=[t??Nn(a)],d=this.runWebGLProgram(c,[{shape:a,dtype:i,dataId:e}],i,u,l,t);return{dtype:i,shape:o,dataId:d.dataId}}runWebGLProgram(e,t,s,r,o=!1,i){const a=this.makeTensorInfo(e.outputShape,s),c=this.texData.get(a.dataId);if(e.packedOutput&&(c.isPacked=!0),e.outPackingScheme===fn.DENSE){const m=i??Nn(e.outputShape);c.texShape=m.map(C=>C*2)}if(e.outTexUsage!=null&&(c.usage=e.outTexUsage),E(a.shape)===0)return c.values=pt(a.dtype,0),a;const l=[],u=t.map(m=>{if(m.dtype==="complex64")throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");let C=this.texData.get(m.dataId);if(C.texture==null){if(!e.packedInputs&&E(m.shape)<=y().getNumber("WEBGL_SIZE_UPLOAD_UNIFORM"))return{shape:m.shape,texData:null,isUniform:!0,uniformValues:C.values};e.packedInputs&&(C.isPacked=!0,C.shape=m.shape)}if(this.uploadToGPU(m.dataId),!!C.isPacked!=!!e.packedInputs)m=C.isPacked?this.unpackTensor(m):this.packTensor(m),l.push(m),C=this.texData.get(m.dataId);else if(C.isPacked&&!jn(C.shape,m.shape)){const b=m,w=m.shape;m.shape=C.shape,m=this.packedReshape(m,w),l.push(m),C=this.texData.get(m.dataId),b.shape=w}return{shape:m.shape,texData:C,isUniform:!1}});this.uploadToGPU(a.dataId);const d={shape:a.shape,texData:c,isUniform:!1},h=Mm(e,u,d),f=this.getAndSaveBinary(h,()=>Lm(this.gpgpu,e,u,d)),p=this.activeTimers!=null;let x;p&&(x=this.startTimer()),y().get("ENGINE_COMPILE_ONLY")||Bm(this.gpgpu,f,u,d,r),l.forEach(m=>this.disposeIntermediateTensorInfo(m)),p&&(x=this.endTimer(x),this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime(x)}));const g=y().getNumber("WEBGL_FLUSH_THRESHOLD");if(g>0){const m=Ae();m-this.lastGlFlushTime>g&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=m)}if(!y().getBool("WEBGL_LAZILY_UNPACK")&&c.isPacked&&o===!1){const m=this.unpackTensor(a);return this.disposeIntermediateTensorInfo(a),m}return a}compileAndRun(e,t,s,r,o=!1){return s=s||t[0].dtype,this.runWebGLProgram(e,t,s,r,o)}getAndSaveBinary(e,t){return e in this.binaryCache||(this.binaryCache[e]=t()),this.binaryCache[e]}getTextureManager(){return this.textureManager}dispose(){this.disposed||(y().getBool("IS_TEST")||Object.keys(this.binaryCache).forEach(t=>{this.gpgpu.deleteProgram(this.binaryCache[t].webGLProgram),delete this.binaryCache[t]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<"u"&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),this.disposed=!0)}floatPrecision(){return this.floatPrecisionValue==null&&(this.floatPrecisionValue=X(()=>{if(!y().get("WEBGL_RENDER_FLOAT32_ENABLED")){const e=y().getBool("DEBUG");y().set("DEBUG",!1);const t=this.abs(st(1e-8)).dataSync()[0];if(y().set("DEBUG",e),t>0)return 32}return 16})),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?S0:I0}uploadToGPU(e){const t=this.texData.get(e),{shape:s,dtype:r,values:o,texture:i,usage:a,isPacked:c}=t;if(i!=null)return;const l=this.activeTimers!=null;let u;l&&(u=Ae());let d=t.texShape;if(d==null&&(d=Hp(s,c),t.texShape=d),o!=null){const h=Fn(s);let f,p=d[1],x=d[0];const g=o instanceof Uint8Array||o instanceof Uint8ClampedArray;(c||!g)&&([p,x]=Yt(d[0],d[1])),c?f=new Hm(h,g):f=new eo(h,g);const m=g?[x,p]:d,C=this.makeTensorInfo(m,r),b=this.texData.get(C.dataId);g?b.usage=Ce.PIXELS:b.usage=Ce.UPLOAD,b.texShape=m,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(C.dataId),p,x,o);const w=[[x,p]],N=this.runWebGLProgram(f,[C],r,w,!0),T=this.texData.get(N.dataId);t.texShape=T.texShape,t.isPacked=T.isPacked,t.usage=T.usage,y().get("ENGINE_COMPILE_ONLY")?this.disposeData(N.dataId):(t.texture=T.texture,t.values=null,this.texData.delete(N.dataId)),this.disposeIntermediateTensorInfo(C),l&&(this.uploadWaitMs+=Ae()-u)}else{const h=this.acquireTexture(d,a,r,c);t.texture=h}}convertAndCacheOnCPU(e,t){const s=this.texData.get(e),{dtype:r}=s;return t!=null&&(s.values=k0(t,r)),s.values}acquireTexture(e,t,s,r){if(this.numBytesInGPU+=this.computeBytes(e,s),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){const o=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${o} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(e,t,r)}computeBytes(e,t){return e[0]*e[1]*Vn(t)}checkCompileCompletion(){for(const[,e]of Object.entries(this.binaryCache))this.checkCompletion_(e)}async checkCompileCompletionAsync(){const e=[];if(this.gpgpu.parallelCompilationExtension){for(const[,t]of Object.entries(this.binaryCache))e.push(this.checkCompletionAsync_(t));return Promise.all(e)}else{for(const[,t]of Object.entries(this.binaryCache)){const s=new Promise(r=>{try{this.checkCompletion_(t),r(!0)}catch(o){throw o}});e.push(s)}return Promise.all(e)}}async checkCompletionAsync_(e){return this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(e):(await ap(),this.checkCompletionAsync_(e))}checkCompletion_(e){if(this.gpgpu.gl.getProgramParameter(e.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(e.webGLProgram)),this.gpgpu.gl.getShaderParameter(e.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?(Aa(e.source,this.gpgpu.gl.getShaderInfoLog(e.fragmentShader)),new Error("Failed to compile fragment shader.")):new Error("Failed to link vertex and fragment shaders.");return!0}getUniformLocations(){for(const e of Object.values(this.binaryCache)){this.gpgpu.buildVao(e.webGLProgram);const{variablesLocations:t,customUniformLocations:s,infLoc:r,nanLoc:o,outShapeLocation:i,outShapeStridesLocation:a,outTexShapeLocation:c}=_a(this.gpgpu,e.program,e.webGLProgram);e.variablesLocations=t,e.customUniformLocations=s,e.infLoc=r,e.nanLoc=o,e.outShapeLocation=i,e.outShapeStridesLocation=a,e.outTexShapeLocation=c}}createTensorFromGPUData(e,t,s){e.channels=e.channels||"RGBA";const{texture:r,height:o,width:i,channels:a}=e,c=Qe().backend;if(!c.gpgpu.gl.isTexture(r))throw new Error("The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.");const l=c.writeTexture(r,t,s,o,i,a);return Qe().makeTensorFromDataId(l,t,s,c)}}os.nextDataId=0;function k0(n,e){if(e==="float32"||e==="complex64")return n;if(e==="int32"||e==="bool"){const t=e==="int32"?new Int32Array(n.length):new Uint8Array(n.length);for(let s=0;s<t.length;++s)t[s]=Math.round(n[s]);return t}else throw new Error(`Unknown dtype ${e}`)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */bi()&&ih("webgl",()=>new os,2);/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cr=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;class wt{constructor(e,t,s){this.variableNames=["A","B"],this.outputShape=ae(t,s),this.enableShapeUniforms=se(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${e}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Et=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`;class tn{constructor(e,t,s,r=!1){this.variableNames=["A","B"],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=ae(t,s);const o=this.outputShape.length;this.enableShapeUniforms=se(o);let i="";if(r)if(o===0||E(this.outputShape)===1)i=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(i=`
          ${U(o)} coords = getOutputCoords();
        `,o===1)this.enableShapeUniforms?i+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:i+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{const c=ie("coords",o);this.enableShapeUniforms?i+=`
            bool nextRowOutOfBounds =
              (${c[o-2]} + 1) >= outShape[${o} - 2];
            bool nextColOutOfBounds =
              (${c[o-1]} + 1) >= outShape[${o} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:i+=`
            bool nextRowOutOfBounds =
              (${c[o-2]} + 1) >= ${this.outputShape[o-2]};
            bool nextColOutOfBounds =
              (${c[o-1]} + 1) >= ${this.outputShape[o-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${e}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${i}

        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ge(n){const{inputs:e,backend:t}=n,{x:s}=e;return t.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const A0={kernelName:Qs,backendName:"webgl",kernelFunc:ge};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rt(n){const{inputs:e,backend:t}=n,{real:s,imag:r}=e,o=t.makeTensorInfo(s.shape,"complex64"),i=t.texData.get(o.dataId),a=ge({inputs:{x:s},backend:t}),c=ge({inputs:{x:r},backend:t});return i.complexTensorInfos={real:a,imag:c},o}const F0={kernelName:Oo,backendName:"webgl",kernelFunc:rt};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ka="return (a < 0.) ? b * a : a;",Ya=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function D0(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{alpha:o}=s,i=t.makeTensorInfo([],"float32",Xt(o,"float32")),a=y().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new tn(Ya,r.shape,i.shape):new wt(Ka,r.shape,i.shape),c=t.runWebGLProgram(a,[r,i],"float32");return t.disposeIntermediateTensorInfo(i),c}const O0={kernelName:Vo,backendName:"webgl",kernelFunc:D0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qa="return (a < 0.) ? b * a : a;",Za=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function P0(n){const{inputs:e,backend:t}=n,{x:s,alpha:r}=e,o=y().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new tn(Za,s.shape,r.shape):new wt(Qa,s.shape,r.shape);return t.runWebGLProgram(o,[s,r],"float32")}const _0={kernelName:zo,backendName:"webgl",kernelFunc:P0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nn="if (isnan(x)) return x;";function _({opSnippet:n,packedOpSnippet:e,cpuKernelImpl:t,dtype:s}){return({inputs:r,backend:o})=>{const{x:i}=r,a=o,c=s||i.dtype;if(a.shouldExecuteOnCPU([i])&&t!=null){const d=a.texData.get(i.dataId),h=t(d.values,c);return a.makeTensorInfo(i.shape,c,h)}const l=y().getBool("WEBGL_PACK_UNARY_OPERATIONS")&&e!=null;let u;return l?u=new et(i.shape,e):u=new Ue(i.shape,n),a.runWebGLProgram(u,[i],c)}}function te({opSnippet:n,packedOpSnippet:e,checkOutOfBounds:t=!1,supportsComplex:s=!1,cpuKernelImpl:r,dtype:o}){return({inputs:i,backend:a})=>{const{a:c,b:l}=i,u=a;if(s&&c.dtype==="complex64"){const p=u.texData.get(c.dataId),x=u.texData.get(l.dataId),[g,m]=[[p.complexTensorInfos.real,x.complexTensorInfos.real],[p.complexTensorInfos.imag,x.complexTensorInfos.imag]].map(b=>{const[w,$]=b,N={dataId:w.dataId,dtype:w.dtype,shape:c.shape},T={dataId:$.dataId,dtype:$.dtype,shape:l.shape},v=new wt(n,c.shape,l.shape);return u.runWebGLProgram(v,[N,T],ze(w.dtype,$.dtype))}),C=rt({inputs:{real:g,imag:m},backend:u});return u.disposeIntermediateTensorInfo(g),u.disposeIntermediateTensorInfo(m),C}const d=o||ze(c.dtype,l.dtype);if((c.dtype==="string"||l.dtype==="string"||u.shouldExecuteOnCPU([c,l]))&&r!=null){const p=u.texData.get(c.dataId).values,x=u.texData.get(l.dataId).values,g=c.dtype==="string"?Gt(p):p,m=c.dtype==="string"?Gt(x):x,[C,b]=r(c.shape,l.shape,g,m,d),w=u.makeTensorInfo(b,d),$=u.texData.get(w.dataId);return $.values=C,w}const h=y().getBool("WEBGL_PACK_BINARY_OPERATIONS")&&e!=null;let f;return h?f=new tn(e,c.shape,l.shape,t):f=new wt(n,c.shape,l.shape),u.runWebGLProgram(f,[c,l],d)}}function pn(n,e=!1){if(n==="linear")return e?x0:h0;if(n==="relu")return e?b0:p0;if(n==="elu")return e?C0:f0;if(n==="relu6")return e?w0:m0;if(n==="prelu")return e?Za:Qa;if(n==="leakyrelu")return e?Ya:Ka;if(n==="sigmoid")return e?y0:g0;throw new Error(`Activation ${n} has not been implemented for the WebGL backend.`)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ja{constructor(e,t,s,r=!1,o=!1,i=!1,a=null,c=!1,l=!1){this.variableNames=["matrixA","matrixB"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=s,this.enableShapeUniforms=se(this.outputShape.length);const u=r?e[1]:e[2],d=Math.ceil(u/2),h=r?"i * 2, rc.y":"rc.y, i * 2",f=o?"rc.z, i * 2":"i * 2, rc.z",p=r?["a.xxyy","a.zzww"]:["a.xxzz","a.yyww"],x=o?["b.xzxz","b.ywyw"]:["b.xyxy","b.zwzw"];let g="",m="";a&&(c?g=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${a}
        }`:l?g=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${a}
        }`:g=`vec4 activation(vec4 x) {
          ${a}
        }`,m="result = activation(result);");const C=i?"result += getBiasAtOutCoords();":"";i&&this.variableNames.push("bias"),c&&this.variableNames.push("preluActivationWeights"),l&&this.variableNames.push("leakyreluAlpha");let b="rc.x",w="rc.x";e[0]<t[0]?b=`imod(rc.x, ${e[0]})`:t[0]<e[0]&&(w=`imod(rc.x, ${t[0]})`),this.userCode=`
      ${g}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${d}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${b};
        int batchB = ${w};
        for (int i = 0; i < ${d}; i++) {
          vec4 a = getMatrixA(batchA, ${h});
          vec4 b = getMatrixB(batchB, ${f});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${p[0]} * ${x[0]});
          result += (${p[1]} * ${x[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${C}

        ${m}

        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lo={REAL:"return areal * breal - aimag * bimag;",IMAG:"return areal * bimag + aimag * breal;"};class uo{constructor(e,t,s){this.variableNames=["AReal","AImag","BReal","BImag"],this.outputShape=ae(t,s),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${e}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ho="return a * b;";function br(n){const{inputs:e,backend:t}=n,{a:s,b:r}=e,o=ze(s.dtype,r.dtype);if(s.dtype==="complex64"){const a=t.texData.get(s.dataId),c=t.texData.get(r.dataId),l=new uo(lo.REAL,s.shape,r.shape),u=new uo(lo.IMAG,s.shape,r.shape),d=[{dataId:a.complexTensorInfos.real.dataId,dtype:a.complexTensorInfos.real.dtype,shape:s.shape},{dataId:a.complexTensorInfos.imag.dataId,dtype:a.complexTensorInfos.imag.dtype,shape:s.shape},{dataId:c.complexTensorInfos.real.dataId,dtype:c.complexTensorInfos.real.dtype,shape:r.shape},{dataId:c.complexTensorInfos.imag.dataId,dtype:c.complexTensorInfos.imag.dtype,shape:r.shape}],h=t.runWebGLProgram(l,d,"float32"),f=t.runWebGLProgram(u,d,"float32"),p=rt({inputs:{real:h,imag:f},backend:t});return t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),p}if(t.shouldExecuteOnCPU([s,r])){const a=t.texData.get(s.dataId),c=t.texData.get(r.dataId),[l,u]=Px(s.shape,r.shape,a.values,c.values,o),d=t.makeTensorInfo(u,o),h=t.texData.get(d.dataId);return h.values=l,d}let i;return y().getBool("WEBGL_PACK_BINARY_OPERATIONS")?i=new tn(ho,s.shape,r.shape):i=new wt(ho,s.shape,r.shape),t.runWebGLProgram(i,[s,r],o)}const L0={kernelName:Wo,backendName:"webgl",kernelFunc:br};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function B0(n,e,t){const s=[zt(n.shape),...Ht(n.shape)],r={dtype:n.dtype,shape:s,dataId:n.dataId},o=[zt(e),...Ht(e)],i=new qa(o,s),a=!0,c=[s],l=t.runWebGLProgram(i,[r],n.dtype,c,a);return{dataId:l.dataId,shape:e,dtype:l.dtype}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function S(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{shape:o}=s,i=t,a=E(r.shape),c=Ac(o,a),l=E(c);I(a===l,()=>`The new shape (${c}) has ${l} elements and the old shape (${r.shape}) has ${a} elements. The new shape and old shape must have the same number of elements.`);const u=i.texData.get(r.dataId);return u.isPacked&&!jn(r.shape,c)&&!(u.texture!==null&&jn(u.shape,c))?B0(r,c,i):(i.incRef(r.dataId),{dataId:r.dataId,shape:c,dtype:r.dtype})}const M0={kernelName:Xo,backendName:"webgl",kernelFunc:S};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fo{constructor(e,t){this.variableNames=["x"];const{windowSize:s,batchSize:r,inSize:o,outSize:i}=e;this.outputShape=[r,i];const a=Math.floor(s/4)*4,c=s%4;let l="sumValue += dot(values, ones);";if(t!=null){const d=1/t;l=`sumValue += dot(values * ${Mn(d)?d.toPrecision(2):d}, ones);`}let u="";o%s>0&&(u=`
        if (inIdx < 0 || inIdx >= ${o}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        float sumValue = 0.0;

        for (int i = 0; i < ${a}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${l}
        }

        int inIdx = inOffset + ${a};
        if (${c===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${l}
        } else if (${c===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${l}
        } else if (${c===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${l}
        }
        setOutput(sumValue);
      }
    `}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class V0{constructor(e,t){this.variableNames=["x"];const{windowSize:s,batchSize:r,inSize:o,outSize:i}=e;this.outputShape=[r,i];let a="0.0",c="";t==="prod"?a="1.0":t==="min"?(a="1.0 / 1e-20",c="min"):t==="max"&&(a="-1.0 / 1e-20",c="max");let l=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="sum"?l="sumValue":t==="prod"?l="prodValue":t==="all"?l="allValue":t==="any"&&(l="anyValue");const u=Math.floor(s/4)*4,d=s%4;let h=`
      if (${t==="sum"}) {
        sumValue += dot(values, ones);
      } else if (${t==="prod"}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${c}(values, minMaxValue);
        if (${t==="min"} || ${t==="max"}) {
          minMaxValue = ${c}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,f="vec4";t==="all"?(a="1.0",h=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,f="bvec4"):t==="any"&&(a="0.0",h=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,f="bvec4");let p="";o%s>0&&(p=`
        if (inIdx < 0 || inIdx >= ${o}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${a};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${p}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        vec4 minMaxValue = vec4(${a});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${h}
        }

        int inIdx = inOffset + ${u};
        if (${d===1}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${h}
        } else if (${d===2}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${h}
        } else if (${d===3}) {
          ${f} values = ${f}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${h}
        }
        setOutput(${l});
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U0(n){const e=[];for(;e.length===0||e[e.length-1].outSize!==1;){const t=e.length?e[e.length-1].outSize:n[1],s=ss(t);e.push({inSize:t,windowSize:s,outSize:Math.ceil(t/s)})}return e}function Nt(n,e,t,s){const r=U0(n.shape);let o=n;for(let i=0;i<r.length;i++){const{inSize:a,windowSize:c,outSize:l}=r[i];let u,d;t==="mean"?u=i===0?new fo({windowSize:c,inSize:a,batchSize:n.shape[0],outSize:l},a):new fo({windowSize:c,inSize:a,batchSize:n.shape[0],outSize:l}):u=new V0({windowSize:c,inSize:a,batchSize:n.shape[0],outSize:l},t),d=o,o=s.runWebGLProgram(u,[o],e),d.dataId!==n.dataId&&s.disposeIntermediateTensorInfo(d)}return o}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class W0{constructor(e,t){this.variableNames=["A"];const s=new Array(e.length);for(let i=0;i<s.length;i++)s[i]=e[t[i]];this.outputShape=s,this.rank=s.length;const r=U(this.rank),o=G0(t);this.userCode=`
    void main() {
      ${r} resRC = getOutputCoords();
      setOutput(getA(${o}));
    }
    `}}function G0(n){const e=n.length;if(e>6)throw Error(`Transpose for rank ${e} is not yet supported`);const t=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u","resRC.v"],s=new Array(e);for(let r=0;r<n.length;r++)s[n[r]]=t[r];return s.join()}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class z0{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0;const s=new Array(e.length);for(let u=0;u<s.length;u++)s[u]=e[t[u]];if(this.outputShape=s,this.rank=s.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);const r=U(this.rank),o=ja("rc",this.rank),i=new Array(this.rank);for(let u=0;u<t.length;u++)i[t[u]]=o[u];const a=`vec2(${i.slice(-2).join()})`,c=`++${o[this.rank-1]} < ${s[this.rank-1]}`,l=`getChannel(getA(${i.join()}), ${a})`;this.userCode=`
    void main() {
      ${r} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${l};
      if(${c}) {
        result[1] = ${l};
      }
      --${o[this.rank-1]};
      if(++${o[this.rank-2]} < ${s[this.rank-2]}) {
        result[2] = ${l};
        if(${c}) {
          result[3] = ${l};
        }
      }
      setOutput(result);
    }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function is(n,e,t){const s=y().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new z0(n.shape,e):new W0(n.shape,e);return t.runWebGLProgram(s,[n],n.dtype)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function H0(n,e,t,s){const r=e,o=n.shape.length,i=de(r,n.shape);let a=i;const c=Te(a,o),l=c!=null;let u=n;l&&(u=is(n,c,s),a=Ee(a.length,o)),Me("sum",a,o);const[d,h]=He(u.shape,a);let f=d;t&&(f=je(d,i));const p=E(h),g=E(n.shape)/p,m=S({inputs:{x:u},attrs:{shape:[g,p]},backend:s}),C=Js(n.dtype),b=Nt(m,C,"sum",s),w=S({inputs:{x:b},attrs:{shape:f},backend:s});return s.disposeIntermediateTensorInfo(m),s.disposeIntermediateTensorInfo(b),l&&s.disposeIntermediateTensorInfo(u),w}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function as(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o,keepDims:i}=s;return H0(r,o,i,t)}const X0={kernelName:Yo,backendName:"webgl",kernelFunc:as};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ce(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{perm:o}=s,i=t,a=r.shape.length,c=new Array(a);for(let u=0;u<c.length;u++)c[u]=r.shape[o[u]];let l;if(i.shouldExecuteOnCPU([r])){const d=i.texData.get(r.dataId).values,h=xr(d,r.shape,r.dtype,o,c);l=i.makeTensorInfo(c,r.dtype);const f=i.texData.get(l.dataId);f.values=h}else l=is(r,o,i);return l}const j0={kernelName:bd,backendName:"webgl",kernelFunc:ce};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ec=1e3;function Kn({a:n,b:e,transposeA:t,transposeB:s,backend:r,bias:o=null,preluActivationWeights:i=null,leakyreluAlpha:a=0,activation:c=null}){const l=n.shape.length,u=e.shape.length,d=t?n.shape[l-2]:n.shape[l-1],h=s?e.shape[u-1]:e.shape[u-2],f=t?n.shape[l-1]:n.shape[l-2],p=s?e.shape[u-2]:e.shape[u-1],x=n.shape.slice(0,-2),g=e.shape.slice(0,-2),m=E(x),C=E(g),w=ae(n.shape.slice(0,-2),e.shape.slice(0,-2)).concat([f,p]);I(d===h,()=>`Error in matMul: inner shapes (${d}) and (${h}) of Tensors with shapes ${n.shape} and ${e.shape} and transposeA=${t} and transposeB=${s} must match.`);const $=t?[m,d,f]:[m,f,d],N=s?[C,p,h]:[C,h,p],T=S({inputs:{x:n},backend:r,attrs:{shape:$}}),v=S({inputs:{x:e},backend:r,attrs:{shape:N}}),D=[T,v],O=Math.max(m,C),L=t?T.shape[1]:T.shape[2],M=o!=null,fe=i!=null,K=c==="leakyrelu",ne=c!=null?pn(c,!0):null,we=M||fe||K||ne!=null;let ke;if((f===1||p===1)&&L>ec&&we===!1){let Ye=T,kt=v;t&&(Ye=ce({inputs:{x:T},backend:r,attrs:{perm:[0,2,1]}}),D.push(Ye)),s&&(kt=ce({inputs:{x:v},backend:r,attrs:{perm:[0,2,1]}}),D.push(kt));const At=p!==1,Rn=p===1;let ls=Ye;At&&(ls=S({inputs:{x:Ye},backend:r,attrs:{shape:[O,L,1]}}),D.push(ls));const Sc=p===1?2:1;let us=kt;Rn&&(us=S({inputs:{x:kt},backend:r,attrs:{shape:[O,1,L]}}),D.push(us));const $r=br({inputs:{a:ls,b:us},backend:r});ke=as({inputs:{x:$r},backend:r,attrs:{axis:Sc,keepDims:!0}}),D.push($r)}else{const Ye=ze(n.dtype,e.dtype),kt=new Ja($,N,[O,f,p],t,s,M,ne,fe,K),At=[T,v];if(o!=null&&At.push(o),fe&&At.push(i),K){const Rn=r.makeTensorInfo([],"float32",Xt(a,"float32"));At.push(Rn),D.push(Rn)}ke=r.runWebGLProgram(kt,At,Ye)}const re=S({inputs:{x:ke},backend:r,attrs:{shape:w}});D.push(ke);for(const Ye of D)r.disposeIntermediateTensorInfo(Ye);return re}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function q0(n){const{inputs:e,backend:t,attrs:s}=n,{a:r,b:o,bias:i,preluActivationWeights:a}=e,{transposeA:c,transposeB:l,activation:u,leakyreluAlpha:d}=s;return Kn({a:r,b:o,transposeA:c,transposeB:l,backend:t,bias:i,preluActivationWeights:a,leakyreluAlpha:d,activation:u})}const K0={kernelName:Id,backendName:"webgl",kernelFunc:q0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const po="return abs(x);";function Y0(n){const{inputs:e,backend:t}=n,{x:s}=e;if(t.shouldExecuteOnCPU([s])&&s.dtype!=="complex64"){const o=t.texData.get(s.dataId),i=Ha(o.values);return t.makeTensorInfo(s.shape,s.dtype,i)}let r;return y().getBool("WEBGL_PACK_UNARY_OPERATIONS")?r=new et(s.shape,po):r=new Ue(s.shape,po),t.runWebGLProgram(r,[s],s.dtype)}const Q0={kernelName:Do,backendName:"webgl",kernelFunc:Y0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Z0=Ne+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`,J0=_({opSnippet:Z0}),eC={kernelName:Hc,backendName:"webgl",kernelFunc:J0};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tC=Ne+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`,nC=_({opSnippet:tC}),sC={kernelName:Xc,backendName:"webgl",kernelFunc:nC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mo="return a + b;",rC=te({opSnippet:mo,packedOpSnippet:mo,supportsComplex:!0,cpuKernelImpl:px}),oC={kernelName:Ks,backendName:"webgl",kernelFunc:rC};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class iC{constructor(e,t){this.outputShape=[],this.outputShape=e,this.variableNames=t.map((o,i)=>`T${i}`);const s=[];this.variableNames.forEach(o=>{s.push(`float v${o} = get${o}AtOutCoords();`)});const r=this.variableNames.map(o=>`v${o}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        float result = ${r};
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class aC{constructor(e,t){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.variableNames=t.map((o,i)=>`T${i}`);const s=[];this.variableNames.forEach(o=>{s.push(`vec4 v${o} = get${o}AtOutCoords();`)});const r=this.variableNames.map(o=>`v${o}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        vec4 result = ${r};
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ln(n){const{inputs:e,backend:t}=n,s=e;if(s.length===1)return ge({inputs:{x:s[0]},backend:t});if(s.length>y().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER")){const c=Math.floor(s.length/2),l=Ln({inputs:s.slice(0,c),backend:t}),u=Ln({inputs:s.slice(c),backend:t});return Ln({inputs:[l,u],backend:t})}const r=s.map(c=>c.dtype).reduce((c,l)=>ze(c,l)),o=s.map(c=>c.shape),a=y().getBool("WEBGL_PACK")?new aC(s[0].shape,o):new iC(s[0].shape,o);return t.runWebGLProgram(a,s,r)}const cC={kernelName:jc,backendName:"webgl",kernelFunc:Ln};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lC(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o,keepDims:i}=s,a=r.shape.length,c=de(o,r.shape);let l=c;const u=Te(l,a);let d=r;u!=null&&(d=ce({inputs:{x:r},backend:t,attrs:{perm:u}}),l=Ee(l.length,a)),Me("all",l,a);const[h,f]=He(d.shape,l),p=E(f),x=S({inputs:{x:d},backend:t,attrs:{shape:[-1,p]}}),g=Nt(x,x.dtype,"all",t);let m;if(i){const C=je(h,c);m=S({inputs:{x:g},backend:t,attrs:{shape:C}})}else m=S({inputs:{x:g},backend:t,attrs:{shape:h}});return t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),u!=null&&t.disposeIntermediateTensorInfo(d),m}const uC={kernelName:qc,backendName:"webgl",kernelFunc:lC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dC(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o,keepDims:i}=s,a=r.shape.length,c=de(o,r.shape);let l=c;const u=Te(l,a);let d=r;u!=null&&(d=ce({inputs:{x:r},backend:t,attrs:{perm:u}}),l=Ee(l.length,a)),Me("any",l,a);const[h,f]=He(d.shape,l),p=E(f),x=S({inputs:{x:d},backend:t,attrs:{shape:[-1,p]}}),g=Nt(x,x.dtype,"any",t);let m;if(i){const C=je(h,c);m=S({inputs:{x:g},backend:t,attrs:{shape:C}})}else m=S({inputs:{x:g},backend:t,attrs:{shape:h}});return t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),u!=null&&t.disposeIntermediateTensorInfo(d),m}const hC={kernelName:Kc,backendName:"webgl",kernelFunc:dC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fC{constructor(e,t,s){this.variableNames=["A"];const{windowSize:r,batchSize:o,outSize:i}=e;s||this.variableNames.push("bestIndicesA"),this.outputShape=[o,i];const a=t==="max"?">":"<",c=s?"inOffset + i;":"round(getBestIndicesA(batch, inOffset + i));";this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${r};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${r}; i++) {
          int inIdx = ${c};
          float candidate = getA(batch, inIdx);
          if (candidate ${a} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class pC{constructor(e,t,s,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,I(e.length>2,()=>`Packed arg${s.charAt(0).toUpperCase()+s.slice(1)} supports only inputs with rank above 2.`);const o=e[e.length-1],i=Math.ceil(o/t);this.outputShape=e.slice(0,-1),i>1&&this.outputShape.push(i),r||this.variableNames.push("bestIndicesA");const a=this.outputShape,c=a.length,l=U(c),u=ie("coords",c);let d,h;if(i===1){h=c+1;const v=U(h);d=`
        ${v} sourceLocR = ${v}(${u.join()}, 0);
        ++${u[c-1]};
        ${v} sourceLocG = ${v}(${u.join()}, 0);
        ++${u[c-2]};
        ${v} sourceLocA = ${v}(${u.join()}, 0);
        --${u[c-1]};
        ${v} sourceLocB = ${v}(${u.join()}, 0);
        --${u[c-2]};`}else h=c,d=`
        ${l} sourceLocR = coords;
        ++${u[c-1]};
        ${l} sourceLocG = coords;
        ++${u[c-2]};
        ${l} sourceLocA = coords;
        --${u[c-1]};
        ${l} sourceLocB = coords;
        --${u[c-2]};`;const f=["x","y","z","w","u","v"].slice(0,h),p="."+f[h-1],x=f.map(v=>"int "+v),g=ie("sourceLocR",h-1).concat("inIdx.r"),m=ie("sourceLocG",h-1).concat("inIdx.g"),C=ie("sourceLocB",h-1).concat("inIdx.b"),b=ie("sourceLocA",h-1).concat("inIdx.a"),w=s==="max"?"greaterThan":"lessThan",$=r?"":`
          inIdx = round(vec4(getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${m.join()}),
                             getBestIndicesAChannel(${C.join()}),
                             getBestIndicesAChannel(${b.join()})));`,N=`vec4(
            getAChannel(${g.join()}),
            hasNextCol ? getAChannel(${m.join()}) : 0.,
            hasNextRow ? getAChannel(${C.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${b.join()}) : 0.)`,T=r?"":`
      float getBestIndicesAChannel(${x.join()}) {
        return getChannel(getBestIndicesA(${f.join()}),
                                          vec2(${f.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${x.join()}) {
        return getChannel(getA(${f.join()}),
                               vec2(${f.slice(-2).join()}));
      }
      ${T}
      void main() {
        ${l} coords = getOutputCoords();
        bool hasNextCol = ${u[c-1]} < ${a[c-1]-1};
        bool hasNextRow = ${u[c-2]} < ${a[c-2]-1};
        ${d}
        ivec4 srcIdx = ivec4(sourceLocR${p}, sourceLocG${p},
          sourceLocB${p}, sourceLocA${p}) * ${t};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${N};

        for (int i = 0; i < ${t}; i++) {
          inIdx = srcIdx;
          ${$}
          vec4 candidate = ${N};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${w}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tc(n,e,t,s=null){let r=e.shape[0],o=e.shape[1];s!=null&&(r=s.shape[0],o=s.shape[1]);const i=ss(o),a={windowSize:i,inSize:o,batchSize:r,outSize:Math.ceil(o/i)},c=new fC(a,t,s==null),l=[e];s!=null&&l.push(s);const u=n.runWebGLProgram(c,l,"int32");if(u.shape[1]===1)return u;const d=tc(n,e,t,u);return n.disposeIntermediateTensorInfo(u),d}function nc(n,e,t,s=null){const r=s!=null?s.shape:e.shape,o=r[r.length-1],i=ss(o),a=new pC(r,i,t,s==null),c=s==null?[e]:[e,s],l=n.runWebGLProgram(a,c,"int32");if(l.shape.length===e.shape.length){const u=nc(n,e,t,l);return n.disposeIntermediateTensorInfo(l),u}return l}function sc(n,e,t,s){const r=[t];if(Me("arg"+s.charAt(0).toUpperCase()+s.slice(1),r,e.shape.length),!y().getBool("WEBGL_PACK_REDUCE")||e.shape.length<=2){const o=[],i=n.texData.get(e.dataId),a=i!==null&&i.isPacked;let c=e;a&&(c=n.unpackTensor(e),o.push(c));const[l,u]=He(c.shape,r),d=E(u),h=S({inputs:{x:c},backend:n,attrs:{shape:[-1,d]}});o.push(h);const f=tc(n,h,s);o.push(f);const p=S({inputs:{x:f},backend:n,attrs:{shape:l}});return o.forEach(x=>n.disposeIntermediateTensorInfo(x)),p}return nc(n,e,s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mC(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o}=s;let i=de(o,r.shape);const a=Te(i,r.shape.length);let c=r;const l=[];a!=null&&(c=ce({inputs:{x:r},backend:t,attrs:{perm:a}}),l.push(c),i=Ee(i.length,c.shape.length)),Me("argMax",[i[0]],c.shape.length);const u=sc(t,c,i[0],"max");return l.forEach(d=>t.disposeIntermediateTensorInfo(d)),u}const gC={kernelName:Yc,backendName:"webgl",kernelFunc:mC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xC(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o}=s;let i=de(o,r.shape);const a=Te(i,r.shape.length);let c=r;const l=[];a!=null&&(c=ce({inputs:{x:r},backend:t,attrs:{perm:a}}),l.push(c),i=Ee(i.length,c.shape.length)),Me("argMin",[i[0]],c.shape.length);const u=sc(t,c,i[0],"min");return l.forEach(d=>t.disposeIntermediateTensorInfo(d)),u}const CC={kernelName:Qc,backendName:"webgl",kernelFunc:xC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bC=Ne+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`,wC=_({opSnippet:bC}),yC={kernelName:Zc,backendName:"webgl",kernelFunc:wC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $C=Ne+"return log(x + sqrt(x * x + 1.0));",vC=_({opSnippet:$C}),SC={kernelName:Jc,backendName:"webgl",kernelFunc:vC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const IC=Ne+`
  return atan(x);
`,RC=_({opSnippet:IC}),TC={kernelName:el,backendName:"webgl",kernelFunc:RC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const EC=Cr+`
  return atan(a, b);
`,NC=`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Et+`
  return result;
`,kC=te({opSnippet:EC,packedOpSnippet:NC}),AC={kernelName:nl,backendName:"webgl",kernelFunc:kC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const FC=Ne+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`,DC=_({opSnippet:FC}),OC={kernelName:tl,backendName:"webgl",kernelFunc:DC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class mn{constructor(e,t,s,r=!1,o=!1){if(this.variableNames=["x"],t==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const i=e.filterWidth,a=e.strideHeight,c=e.strideWidth,l=e.dilationHeight,u=e.dilationWidth,d=e.effectiveFilterHeight,h=e.effectiveFilterWidth,f=e.padInfo.top,p=e.padInfo.left;this.outputShape=e.outShape;const x=t==="avg",g=`((batch  * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + d`,m=`(xR * ${e.inWidth} + xC) * ${e.inChannels} + d`;let C="0.0";if(x||(C="-1.0 / 1e-20"),s){const v=">=";this.userCode=`
        const ivec2 strides = ivec2(${a}, ${c});
        const ivec2 pads = ivec2(${f}, ${p});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${d};
              wR += ${l}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${h};
                wC += ${u}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value ${v} currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${r?o?g:m:`wR * ${h} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const b="max";let w=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="avg"&&(w="avgValue / max(count, 1.0)");const $=Math.floor(i/4)*4,N=i%4,T=`
      if (${x}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${b}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${a}, ${c});
      const ivec2 pads = ivec2(${f}, ${p});
      const float initializationValue = ${C};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${C});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${d};
            wR += ${l}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${$}; wC += 4) {
            int xC = xCCorner + wC * ${u};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              getValue(batch, xR, xC + 3 * ${u}, d)
            );

            ${T}
          }

          int xC = xCCorner + ${$};
          if (${N===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${T}
          } else if (${N===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              initializationValue,
              initializationValue
            );

            ${T}
          } else if (${N===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              initializationValue
            );

            ${T}
          }
        }
        setOutput(${w});
      }
    `}}class wr{constructor(e,t,s,r=!1,o=!1){if(this.variableNames=["x"],t==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const i=e.filterWidth,a=e.strideDepth,c=e.strideHeight,l=e.strideWidth,u=e.dilationDepth,d=e.dilationHeight,h=e.dilationWidth,f=e.effectiveFilterDepth,p=e.effectiveFilterHeight,x=e.effectiveFilterWidth,g=e.padInfo.front,m=e.padInfo.top,C=e.padInfo.left;this.outputShape=e.outShape;const b=t==="avg";let w="0.0";if(b||(w="-1.0 / 1e-20"),s){const O=">=";this.userCode=`
        const ivec3 strides =
            ivec3(${a}, ${c}, ${l});
        const ivec3 pads = ivec3(${g}, ${m}, ${C});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${f};
              wD += ${u}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${e.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${p};
                wR += ${d}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${x};
                  wC += ${h}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value ${O} currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${r?o?`(((batch * ${e.inDepth} + xD) * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`((xD * ${e.inHeight} + xR) * ${e.inWidth} + xC) * ${e.inChannels} + ch`:`wD * ${p} * ${x} +
                      wR * ${x} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const $="max";let N=`${t}(${t}(${t}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;t==="avg"&&(N="avgValue / max(count, 1.0)");const T=Math.floor(i/4)*4,v=i%4,D=`
      if (${b}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${$}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${a}, ${c}, ${l});
      const ivec3 pads = ivec3(${g}, ${m}, ${C});
      const float initializationValue = ${w};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${e.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${w});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${f};
            wD += ${u}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${p};
            wR += ${d}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${T}; wC += 4) {
              int xC = xCCorner + wC * ${h};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${h}, ch),
                getValue(batch, xD, xR, xC + 2 * ${h}, ch),
                getValue(batch, xD, xR, xC + 3 * ${h}, ch)
              );

              ${D}
            }

            int xC = xCCorner + ${T};
            if (${v===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${D}
            } else if (${v===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${h}, ch),
                initializationValue,
                initializationValue
              );

              ${D}
            } else if (${v===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${h}, ch),
                getValue(batch, xD, xR, xC + 2 * ${h}, ch),
                initializationValue
              );

              ${D}
            }
          }
        }
        setOutput(${N});
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function PC(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e;$n(r,"avgPool");const{filterSize:o,strides:i,pad:a,dimRoundingMode:c}=s,l=1;I(qt(i,l),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${l}'`);const u=jt(r.shape,o,i,l,a,c);if(u.filterWidth===1&&u.filterHeight===1&&J(u.inShape,u.outShape))return ge({inputs:{x:r},backend:t});const d=new mn(u,"avg",!1);return t.runWebGLProgram(d,[r],"float32")}const _C={kernelName:sl,backendName:"webgl",kernelFunc:PC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function LC(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{filterSize:o,strides:i,pad:a,dimRoundingMode:c,dataFormat:l}=s,u=[1,1,1],d=bn(r.shape,o,i,u,a,c,l),h=new wr(d,"avg",!1);return t.runWebGLProgram(h,[r],"float32")}const BC={kernelName:ol,backendName:"webgl",kernelFunc:LC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class MC{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,r=e.strideHeight,o=e.strideWidth,i=e.dilationHeight,a=e.dilationWidth,c=e.effectiveFilterHeight,l=e.effectiveFilterWidth,u=c-1-e.padInfo.top,d=l-1-e.padInfo.left,h=1/(t*s);this.userCode=`
      const ivec2 pads = ivec2(${u}, ${d});
      const float avgMultiplier = float(${h});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${c};
            wR += ${i}) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${l};
            wC+= ${a}) {
            float dyC = float(dyCCorner + wC) / ${o}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}}class VC{constructor(e){this.variableNames=["dy"],this.outputShape=e.inShape;const t=e.filterDepth,s=e.filterHeight,r=e.filterWidth,o=e.strideDepth,i=e.strideHeight,a=e.strideWidth,c=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,d=e.effectiveFilterDepth,h=e.effectiveFilterHeight,f=e.effectiveFilterWidth,p=d-1-e.padInfo.front,x=h-1-e.padInfo.top,g=f-1-e.padInfo.left,m=1/(t*s*r);this.userCode=`
      const ivec3 pads = ivec3(${p}, ${x}, ${g});
      const float avgMultiplier = float(${m});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${d};
            wD += ${c}) {
          float dyD = float(dyDCorner + wD) / ${o}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${h};
              wR += ${l}) {
            float dyR = float(dyRCorner + wR) / ${i}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${f};
                wC += ${u}) {
              float dyC = float(dyCCorner + wC) / ${a}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function UC(n){const{inputs:e,backend:t,attrs:s}=n,{dy:r,input:o}=e,i=o,{filterSize:a,strides:c,pad:l,dimRoundingMode:u}=s,d=[1,1,1],h=bn(i.shape,a,c,d,l,u),f=new VC(h);return t.runWebGLProgram(f,[r],i.dtype)}const WC={kernelName:il,backendName:"webgl",kernelFunc:UC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function GC(n){const{inputs:e,backend:t,attrs:s}=n,{dy:r,input:o}=e,i=o;$n([r,o],"avgPoolGrad");const{filterSize:a,strides:c,pad:l}=s,u=jt(i.shape,a,c,1,l),d=new MC(u);return t.runWebGLProgram(d,[r],i.dtype)}const zC={kernelName:rl,backendName:"webgl",kernelFunc:GC};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function HC(n){const{inputs:e,backend:t,attrs:s}=n,{a:r,b:o}=e,{transposeA:i,transposeB:a}=s;return Kn({a:r,b:o,transposeA:i,transposeB:a,backend:t})}const XC={kernelName:al,backendName:"webgl",kernelFunc:HC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class jC{constructor(e,t,s,r,o,i){this.outputShape=[],this.variableNames=["x","mean","variance"],ae(e,t),ae(e,s);let a="0.0";r!=null&&(ae(e,r),this.variableNames.push("offset"),a="getOffsetAtOutCoords()");let c="1.0";o!=null&&(ae(e,o),this.variableNames.push("scale"),c="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${a};
        float scale = ${c};
        float inv = scale * inversesqrt(variance + float(${i}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class qC{constructor(e,t,s,r,o,i){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=["x","mean","variance"],ae(e,t),ae(e,s);let a="vec4(0.0)";r!=null&&(ae(e,r),this.variableNames.push("offset"),a="getOffsetAtOutCoords()");let c="vec4(1.0)";o!=null&&(ae(e,o),this.variableNames.push("scale"),c="getScaleAtOutCoords()"),this.outputShape=e,this.userCode=`
      void main() {
        vec4 offset = ${a};
        vec4 scale = ${c};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${i}));

        setOutput((x - mean) * inv + offset);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const KC=({inputs:n,backend:e,attrs:t})=>{const{x:s,mean:r,variance:o,offset:i,scale:a}=n;I(r.shape.length===o.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),I(i==null||r.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),I(a==null||r.shape.length===a.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");let{varianceEpsilon:c}=t;c==null&&(c=.001);const l=[s,r,o];let u=null;i!=null&&(u=i.shape,l.push(i));let d=null;a!=null&&(d=a.shape,l.push(a));const h=y().getBool("WEBGL_PACK_NORMALIZATION")?new qC(s.shape,r.shape,o.shape,u,d,c):new jC(s.shape,r.shape,o.shape,u,d,c);return e.runWebGLProgram(h,l,l[0].dtype)},YC={kernelName:Gl,backendName:"webgl",kernelFunc:KC};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class QC{constructor(e){this.variableNames=["source"],this.outputShape=e,this.rank=e.length;const t=U(this.rank);this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const s=ZC(this.rank);let r;const o=e.map((i,a)=>`sourceLoc.${Ws[a]} = start[${a}] + coords.${Ws[a]};`);r=`
        ${t} sourceLoc;
        ${t} coords = getOutputCoords();
        ${o.join(`
`)}
      `,this.userCode=`
      void main() {
        ${r}
        setOutput(getSource(${s}));
      }
    `}}const Ws=["x","y","z","w","u","v"];function ZC(n){if(n===1)return"sourceLoc";if(n<=6)return Ws.slice(0,n).map(e=>"sourceLoc."+e).join(",");throw Error(`Slicing for rank ${n} is not yet supported`)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class JC{constructor(e){this.variableNames=["source"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e,this.rank=e.length,this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const t=U(this.rank),s=ie("coords",this.rank),r=ie("sourceLoc",this.rank),o=this.rank===1?"sourceLoc":`vec2(${r.slice(-2).join()})`,i=`getChannel(getSource(${r.join()}), ${o})`,a=`
      result.x = ${i};
      if (++${s[this.rank-1]} < ${e[this.rank-1]}) {
        ++${r[this.rank-1]};
        result.y = ${i};
        --${r[this.rank-1]};
      }
    `,c=this.rank===1?"":`
      --${s[this.rank-1]};
      if (++${s[this.rank-2]} < ${e[this.rank-2]}) {
        ++${r[this.rank-2]};
        result.z = ${i};
        if (++${s[this.rank-1]} < ${e[this.rank-1]}) {
          ++${r[this.rank-1]};
          result.w = ${i};
        }
      }
    `,l=this.rank<=4?`sourceLoc = coords +
            ${t}(${e.map((u,d)=>`start[${d}]`).join()});`:e.map((u,d)=>`${r[d]} = ${s[d]} + start[${d}];`).join(`
`);this.userCode=`
      void main() {
        ${t} coords = getOutputCoords();
        ${t} sourceLoc;
        ${l}
        vec4 result = vec4(0.);
        ${a}
        ${c}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eb(n,e,t,s){const r=s.texData.get(n.dataId),o=s.makeTensorInfo(t,n.dtype),i=s.texData.get(o.dataId);Object.assign(i,r),i.refCount=1,i.shape=t,i.dtype=n.dtype;let a=cr(e,Z(n.shape));r.slice&&(a+=r.slice.flatOffset),i.slice={flatOffset:a,origDataId:r.slice&&r.slice.origDataId||n.dataId};const c=s.dataRefCount.get(i.slice.origDataId)||1;return s.dataRefCount.set(i.slice.origDataId,c+1),o}function sn(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{begin:o,size:i}=s,[a,c]=Xi(r,o,i);if(_i(r,a,c),E(c)===0)return t.makeTensorInfo(c,r.dtype,[]);if(t.shouldExecuteOnCPU([r])||r.dtype==="string"){const d=t.texData.get(r.dataId),h=Xx(d.values,a,c,r.shape,r.dtype);return t.makeTensorInfo(c,r.dtype,h)}const{isPacked:l}=t.texData.get(r.dataId),u=ar(r.shape,a,c);if(l||!u){const d=y().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new JC(c):new QC(c),h=[a];return t.runWebGLProgram(d,[r],r.dtype,h)}return t.uploadToGPU(r.dataId),eb(r,a,c,t)}const tb={kernelName:Ku,backendName:"webgl",kernelFunc:sn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nb=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{blockShape:o,crops:i}=s;I(r.shape.length<=4,()=>"batchToSpaceND for rank > 4 with a WebGL backend not implemented yet");const a=o.reduce((C,b)=>C*b),c=ur(r.shape,o,a),l=dr(c.length,o.length),u=hr(r.shape,o,a),d=ea(i,o.length),h=ta(u,i,o.length),f=[],p=S({inputs:{x:r},backend:t,attrs:{shape:c}}),x=ce({inputs:{x:p},backend:t,attrs:{perm:l}}),g=S({inputs:{x},backend:t,attrs:{shape:u}}),m=sn({inputs:{x:g},backend:t,attrs:{begin:d,size:h}});return f.push(p),f.push(x),f.push(g),f.forEach(C=>t.disposeIntermediateTensorInfo(C)),m},sb={kernelName:cl,backendName:"webgl",kernelFunc:nb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rb(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,weights:o}=e,{size:i}=s,a=t.readSync(r.dataId),c=t.readSync(o.dataId),l=za(a,c,o.dtype,o.shape,i);return t.makeTensorInfo([i],o.dtype,l)}const ob={kernelName:ll,backendName:"webgl",kernelFunc:rb};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ib=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,ab=`
  return float(int(a.r) & int(b.r));
`;function cb(n){const{inputs:e,backend:t}=n,{a:s,b:r}=e,o=y().getBool("WEBGL_PACK_BINARY_OPERATIONS"),i=y().getNumber("WEBGL_VERSION");if(t.shouldExecuteOnCPU([s,r])||i===1){const c=t.texData.get(s.dataId).values,l=t.texData.get(r.dataId).values,[u,d]=gx(s.shape,r.shape,c,l,s.dtype),h=t.makeTensorInfo(d,s.dtype),f=t.texData.get(h.dataId);return f.values=u,h}let a;return o?a=new tn(ib,s.shape,r.shape,!1):a=new wt(ab,s.shape,r.shape),t.runWebGLProgram(a,[s,r],s.dtype)}const lb={kernelName:ul,backendName:"webgl",kernelFunc:cb};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ub(n){const{inputs:e,backend:t}=n,{s0:s,s1:r}=e,o=t.readSync(s.dataId),i=t.readSync(r.dataId),a=ae(Array.from(o),Array.from(i));return t.makeTensorInfo([a.length],"int32",Int32Array.from(a))}const db={kernelName:dl,backendName:"webgl",kernelFunc:ub};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hb="return float(a != b);",rc=te({opSnippet:hb,cpuKernelImpl:Lx,dtype:"bool"}),fb={kernelName:$u,backendName:"webgl",kernelFunc:rc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sn(n){const{inputs:e,backend:t}=n,{input:s}=e,r=t.texData.get(s.dataId);return ge({inputs:{x:r.complexTensorInfos.real},backend:t})}const pb={kernelName:Pu,backendName:"webgl",kernelFunc:Sn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mb="return float(int(x));";function gb(n,e){const t=new Ue(n.shape,mb),s=e.runWebGLProgram(t,[n],"int32");return{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gs(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{dtype:o}=s;if(o==="complex64"){if(r.dtype==="complex64")return ge({inputs:{x:r},backend:t});const i=Ls(r.shape),a=Gs({inputs:{x:r},backend:t,attrs:{dtype:"float32"}}),c=rt({inputs:{real:a,imag:i},backend:t});return i.dispose(),t.disposeIntermediateTensorInfo(a),c}if(r.dtype==="complex64"){const i=Sn({inputs:{input:r},backend:t}),a=Gs({inputs:{x:i},backend:t,attrs:{dtype:o}});return t.disposeIntermediateTensorInfo(i),a}if(!Oc(r.dtype,o)){const i=ge({inputs:{x:r},backend:t});return{dataId:i.dataId,shape:i.shape,dtype:o}}if(t.shouldExecuteOnCPU([r])){const i=t.texData.get(r.dataId).values,[a,c,l]=xx(i,r.shape,r.dtype,o);return t.makeTensorInfo(a,c,l)}if(o==="int32")return gb(r,t);if(o==="bool"){const i=t.makeTensorInfo([],"bool",pt("bool",1)),c=rc({inputs:{a:r,b:i},backend:t});return t.disposeIntermediateTensorInfo(i),c}throw new Error(`Error in Cast: failed to cast ${r.dtype} to ${o}`)}const xb={kernelName:Ys,backendName:"webgl",kernelFunc:Gs};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const go="return ceil(x);",Cb=_({opSnippet:go,packedOpSnippet:go,cpuKernelImpl:Cx}),bb={kernelName:hl,backendName:"webgl",kernelFunc:Cb};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wb{constructor(e){this.variableNames=["A"],this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class yb{constructor(e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=e,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $b(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{clipValueMin:o,clipValueMax:i}=s;let a;y().getBool("WEBGL_PACK_CLIP")?a=new yb(r.shape):a=new wb(r.shape);const c=[[o],[i]];return t.runWebGLProgram(a,[r],r.dtype,c)}const vb={kernelName:fl,backendName:"webgl",kernelFunc:$b};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Sb{constructor(e){this.variableNames=["real","imag"],this.outputShape=e,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xo(n,e){return{dataId:e.dataId,dtype:e.dtype,shape:n.shape}}function Ib(n){const{inputs:e,backend:t}=n,{x:s}=e,r=t.texData.get(s.dataId),o=new Sb(s.shape),i=[xo(s,r.complexTensorInfos.real),xo(s,r.complexTensorInfos.imag)];return t.runWebGLProgram(o,i,i[0].dtype)}const Rb={kernelName:Po,backendName:"webgl",kernelFunc:Ib};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Tb{constructor(e){this.outputShape=[],this.outputShape=bt(e,1),this.variableNames=e.map((i,a)=>`T${a}`);const t=new Array(e.length-1);t[0]=e[0][1];for(let i=1;i<t.length;i++)t[i]=t[i-1]+e[i][1];const s=[`if (yC < ${t[0]}) setOutput(getT0(yR, yC));`];for(let i=1;i<t.length;i++){const a=t[i-1];s.push(`else if (yC < ${t[i]}) setOutput(getT${i}(yR, yC-${a}));`)}const r=t.length,o=t[t.length-1];s.push(`else setOutput(getT${r}(yR, yC-${o}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${s.join(`
        `)}
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Eb{constructor(e,t){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=bt(e,t);const s=this.outputShape,r=s.length,o=U(r),i=ie("coords",r),a=["x","y","z","w","u","v"].slice(0,r);this.variableNames=e.map((x,g)=>`T${g}`);const c=new Array(e.length-1);c[0]=e[0][t];for(let x=1;x<c.length;x++)c[x]=c[x-1]+e[x][t];const l=a[t],u=a.slice(-2),d=a.join();let h=`if (${l} < ${c[0]}) {
        return getChannel(
            getT0(${d}), vec2(${u.join()}));
        }`;for(let x=1;x<c.length;x++){const g=c[x-1];h+=`
        if (${l} < ${c[x]}  && ${l} >= ${c[x-1]}) {
          return getChannel(
            getT${x}(${Pn(a,l,g)}),
            vec2(${Pn(u,l,g)}));
        }`}const f=c.length,p=c[c.length-1];h+=`
        return getChannel(
          getT${f}(${Pn(a,l,p)}),
          vec2(${Pn(u,l,p)}));`,this.userCode=`
      float getValue(${a.map(x=>"int "+x)}) {
        ${h}
      }

      void main() {
        ${o} coords = getOutputCoords();
        vec4 result = vec4(getValue(${i}), 0., 0., 0.);

        ${i[r-1]} = ${i[r-1]} + 1;
        if (${i[r-1]} < ${s[r-1]}) {
          result.g = getValue(${i});
        }

        ${i[r-2]} = ${i[r-2]} + 1;
        if (${i[r-2]} < ${s[r-2]}) {
          result.a = getValue(${i});
        }

        ${i[r-1]} = ${i[r-1]} - 1;
        if (${i[r-2]} < ${s[r-2]} &&
            ${i[r-1]} < ${s[r-1]}) {
          result.b = getValue(${i});
        }
        setOutput(result);
      }
    `}}function Pn(n,e,t){const s=n.indexOf(e);return n.map((o,i)=>i===s?`${o} - ${t}`:o).join()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cs(n){const{inputs:e,backend:t}=n,{input:s}=e,r=t.texData.get(s.dataId);return ge({inputs:{x:r.complexTensorInfos.imag},backend:t})}const Nb={kernelName:Kl,backendName:"webgl",kernelFunc:cs};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function un(n,e,t){const s=n[0].dtype;if(s==="complex64"){const f=n.map(C=>Sn({inputs:{input:C},backend:t})),p=n.map(C=>cs({inputs:{input:C},backend:t})),x=un(f,e,t),g=un(p,e,t),m=rt({inputs:{real:x,imag:g},backend:t});return f.forEach(C=>t.disposeIntermediateTensorInfo(C)),p.forEach(C=>t.disposeIntermediateTensorInfo(C)),t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),m}let r=t.shouldExecuteOnCPU(n);if(s==="string"&&(r=!0),r){const f=n.map(w=>{const N=[-1,E(w.shape.slice(e))];return S({inputs:{x:w},backend:t,attrs:{shape:N}})}),p=f.map(w=>({vals:t.readSync(w.dataId),shape:w.shape})),x=bt(f.map(w=>w.shape),1),g=f[0].shape[0]===1,m=bx(p,x,s,g),C=bt(n.map(w=>w.shape),e),b=t.makeTensorInfo(C,s,m);return f.forEach(w=>t.disposeIntermediateTensorInfo(w)),b}const o=n.filter(f=>E(f.shape)>0),i=y().getBool("WEBGL_PACK_ARRAY_OPERATIONS")&&o[0].shape.length>1;if(o.length===1){const f=i?new Ue(n[0].shape,Ze):new et(n[0].shape,Ze);return t.runWebGLProgram(f,n,s)}const a=y().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER");if(o.length>a){const f=[];for(let x=0;x<o.length;x+=a){const g=o.slice(x,x+a);f.push(un(g,e,t))}const p=un(f,e,t);for(const x of f)t.disposeIntermediateTensorInfo(x);return p}if(i){const f=new Eb(o.map(p=>p.shape),e);return t.runWebGLProgram(f,o,s)}const{tensors2D:c,outShape:l}=kb(o,e,t),u=new Tb(c.map(f=>f.shape)),d=t.runWebGLProgram(u,c,s);c.forEach(f=>t.disposeIntermediateTensorInfo(f));const h=S({inputs:{x:d},attrs:{shape:l},backend:t});return t.disposeIntermediateTensorInfo(d),h}function kb(n,e,t){const s=bt(n.map(o=>o.shape),e);return{tensors2D:n.map(o=>S({inputs:{x:o},attrs:{shape:[-1,E(o.shape.slice(e))]},backend:t})),outShape:s}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oc(n){const{inputs:e,backend:t,attrs:s}=n,{axis:r}=s,o=de(r,e[0].shape)[0],i=e.map(l=>l.shape);qi(i,o);const a=bt(e.map(l=>l.shape),o);if(E(a)===0)return t.makeTensorInfo(a,e[0].dtype,[]);const c=e.filter(l=>E(l.shape)>0);return c.length===1?ge({inputs:{x:c[0]},backend:t}):un(c,o,t)}const Ab={kernelName:pl,backendName:"webgl",kernelFunc:oc};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ic{constructor(e,t=!1,s=null,r=!1,o=!1){this.variableNames=["x","W"],this.outputShape=e.outShape;const i=e.padInfo.top,a=e.padInfo.left,c=e.strideHeight,l=e.strideWidth,u=e.dilationHeight,d=e.dilationWidth,h=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,x=e.inChannels%4,g=e.dataFormat==="channelsLast",m=g?1:2,C=g?2:3,b=g?3:1;let w="",$="";s&&(r?w=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:o?w=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:w=`
          float activation(float x) {
            ${s}
          }
        `,$="result = activation(result);");const N=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),o&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${w}

      const ivec2 strides = ivec2(${c}, ${l});
      const ivec2 pads = ivec2(${i}, ${a});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${b}];

        ivec2 xRCCorner =
            ivec2(coords[${m}], coords[${C}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${h}; wR++) {
          int xR = xRCorner + wR * ${u};

          if (xR < 0 || xR >= ${e.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${f}; wC++) {
            int xC = xCCorner + wC * ${d};

            if (xC < 0 || xC >= ${e.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${p}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${g}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${x===1}) {

              if (${g}) {
                dotProd +=
                    getX(batch, xR, xC, ${p}) *
                    getW(wR, wC, ${p}, d2);
              } else {
                dotProd +=
                    getX(batch, ${p}, xR, xC) *
                    getW(wR, wC, ${p}, d2);
              }

            } else if (${x===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2)
              );

              if (${g}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${x===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${p}, d2),
                getW(wR, wC, ${p} + 1, d2),
                getW(wR, wC, ${p} + 2, d2)
              );

              if (${g}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${p}),
                  getX(batch, xR, xC, ${p} + 1),
                  getX(batch, xR, xC, ${p} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${p}, xR, xC),
                  getX(batch, ${p} + 1, xR, xC),
                  getX(batch, ${p} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${N}
        ${$}
        setOutput(result);
      }
    `}}class Fb{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const t=e.padInfo.front,s=e.padInfo.top,r=e.padInfo.left,o=e.strideDepth,i=e.strideHeight,a=e.strideWidth,c=e.dilationDepth,l=e.dilationHeight,u=e.dilationWidth,d=e.filterDepth,h=e.filterHeight,f=e.filterWidth,p=Math.floor(e.inChannels/4)*4,x=e.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${o}, ${i}, ${a});
      const ivec3 pads = ivec3(${t}, ${s}, ${r});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${d}; wF++) {
          int xF = xFCorner + wF * ${c};

          if (xF < 0 || xF >= ${e.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${h}; wR++) {
            int xR = xRCorner + wR * ${l};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${f}; wC++) {
              int xC = xCCorner + wC * ${u};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${p}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${x===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${p}) *
                  getW(wF, wR, wC, ${p}, d2);
              } else if (${x===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${x===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${p}),
                  getX(batch, xF, xR, xC, ${p} + 1),
                  getX(batch, xF, xR, xC, ${p} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${p}, d2),
                  getW(wF, wR, wC, ${p} + 1, d2),
                  getW(wF, wR, wC, ${p} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ac{constructor(e,t=!1,s=null,r=!1,o=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=se(this.outputShape.length);const i=e.padInfo.left,a=e.strideWidth,c=e.dilationWidth,l=e.filterHeight,u=e.filterWidth,d=u;let h=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let g=0;g<u;g++)h+=`
           vec4 xTexelC${g*2};
           int xTexelC${g*2}Ready;
           vec4 xTexelC${g*2+1};
           int xTexelC${g*2+1}Ready;
           vec4 xC${g};`;h+=`
     for (int r = 0; r < ${l}; r++) {
      for (int d1 = 0; d1 < ${e.inChannels}; d1 += 2) {
       `;for(let g=0;g<u;g++)h+=`
           xTexelC${g*2} = vec4(0.0);
           xTexelC${g*2}Ready = 0;
           xTexelC${g*2+1} = vec4(0.0);
           xTexelC${g*2+1}Ready = 0;
           xC${g} = vec4(0.0);`;h+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let g=0;g<(d+1)/2;g++){const m=g*2;if(h+=`
           xC = xCCorner + ${m*c};
           `,a===1){if(m<u&&(i%2===1?(h+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }
               `,c===1&&m>0?h+=`
                 xC${m} = vec4(xTexelC${m-2}.zw, xTexelC${m}.xy);
                 `:h+=`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${m} = vec4(previous.zw, xTexelC${m}.xy);
                   } else {
                     xC${m} = vec4(0.0, 0.0, xTexelC${m}.xy);
                   }
                   `):h+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 xC${m} = xTexelC${m};
                 `,m+1<u)){const C=i%2===0?Hs(c):c;c%2===0&&i%2===1||c%2!==0&&i%2!==1?(h+=`
                   xCOffset = xC + imod(pads[1], 2) + ${C};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                     xTexelC${m+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${m+1}.zw = vec2(0.0);
                     }
                     xTexelC${m+1}Ready = 1;
                   }
                   `,c>1?h+=`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${m+1} = vec4(previous.zw, xTexelC${m+1}.xy);
                     } else {
                      xC${m+1} = vec4(0.0, 0.0, xTexelC${m+1}.xy);
                     }
                     `:h+=`
                     xC${m+1} = vec4(xTexelC${m}.zw, xTexelC${m+1}.xy);
                     `):C===1?h+=`
                     xC${m+1} = xTexelC${m};
                     `:h+=`
                     xCOffset = xC + ${C};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                       xTexelC${m+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${m+1}.zw = vec2(0.0);
                       }
                       xTexelC${m+1}Ready = 1;
                     }

                     xC${m+1} = xTexelC${m+1};
                     `}}else m<u&&(i%2===1?(h+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${m+1}Ready == 0) {
                   xTexelC${m+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${m+1}.zw = vec2(0.0);
                   }
                   xTexelC${m+1}Ready = 1;
                 }

                 xC${m} = vec4(xTexelC${m}.zw, xTexelC${m+1}.zw);
               `,m+1<u&&(h+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${m+1} = vec4(xTexelC${m+1}.xy, final.xy);
                 `)):(h+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${m}Ready == 0) {
                   xTexelC${m} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${m}.zw = vec2(0.0);
                   }
                   xTexelC${m}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${m+1}Ready == 0) {
                   xTexelC${m+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${m+1}.zw = vec2(0.);
                   }
                   xTexelC${m+1}Ready = 1;
                 }

                 xC${m} = vec4(
                   xTexelC${m}.xy, xTexelC${m+1}.xy);
               `,m+1<u&&(h+=`
                   xC${m+1} = vec4(xTexelC${m}.zw, xTexelC${m+1}.zw);
                 `)));m<u&&(h+=`
             wTexel = getW(r, ${m}, d1, d2);
             dotProd += xC${m}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${e.inChannels}) {
               dotProd += xC${m}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,m+1<u&&(h+=`
               wTexel = getW(r, ${m+1}, d1, d2);
               dotProd += xC${m+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${e.inChannels}) {
                 dotProd += xC${m+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}h+=`
     }
   `,h+=`
     }
   `,h+=`
     }
   `;let f="",p="";s&&(r?f=`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${s}
         }`:o?f=`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${s}
         }`:f=`vec4 activation(vec4 x) {
           ${s}
         }`,p="result = activation(result);");const x=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),o&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
       ${f}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${h}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${x}
         ${p}
         setOutput(result);
       }
     `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Db{constructor(e,t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec4"},{name:"pad",type:"ivec2"},{name:"stride",type:"ivec2"},{name:"dilation",type:"ivec2"},{name:"inChannels",type:"int"},{name:"itemsPerBlockRow",type:"int"},{name:"outWidth",type:"int"}],this.outputShape=e,this.enableShapeUniforms=se(this.outputShape.length);const{dataFormat:s}=t,r=le(),o=s==="channelsLast",i=o?1:2,a=o?2:3,c=this.enableShapeUniforms?"if(blockIndex < outShape[2] && pos < outShape[1]) {":`if(blockIndex < ${e[2]} && pos < ${e[1]}) {`;let l="";for(let u=0;u<=1;u++)for(let d=0;d<=1;d++)l+=`
          blockIndex = rc.z + ${d};
          pos = rc.y + ${u};

          ${c}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${i}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${a}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${o}) {
                  innerDims = vec2(d1, ch);
                  result[${u*2+d}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${u*2+d}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${l}

        ${r.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yn(n,e){const t=n.length;return t>=3?e?[...n.slice(0,-3),n[t-3]*n[t-2],n[t-1]]:[...n.slice(0,-3),n[t-3],n[t-2]*n[t-1]]:!e&&t===1&&n[0]>1?[n[0],1]:null}function cc({x:n,filter:e,convInfo:t,backend:s,bias:r=null,preluActivationWeights:o=null,leakyreluAlpha:i=0,activation:a=null}){const c=n.shape,l=s.texData.get(n.dataId),u=t.inChannels,d=c[0]*c[1]*c[2],h=t.outChannels,f=t.dataFormat==="channelsLast",p=!1,x=!1;let g;const m=[];if(o!=null){const w=Yn(o.shape,f);w!=null&&(o=S({inputs:{x:o},backend:s,attrs:{shape:w}}),m.push(o))}if(r!=null){const w=Yn(r.shape,f);w!=null&&(r=S({inputs:{x:r},backend:s,attrs:{shape:w}}),m.push(r))}if(!((d===1||h===1)&&u>ec)&&l.isPacked&&f&&l.texture!=null&&c[2]%2!==0&&J(l.shape.slice(-3),c.slice(-3))){const w=c[0]*c[1]*(c[2]+1),$={dataId:n.dataId,shape:[1,w,t.inChannels],dtype:n.dtype},N=l.shape;l.shape=l.shape.slice(),l.shape[l.shape.length-2]++,I(jn(l.shape,$.shape),()=>`packed reshape ${l.shape} to ${$.shape} isn't free`);const T=S({inputs:{x:e},backend:s,attrs:{shape:[1,t.inChannels,t.outChannels]}});m.push(T);const v=Kn({a:$,b:T,backend:s,transposeA:p,transposeB:x,bias:r,activation:a,preluActivationWeights:o,leakyreluAlpha:i}),D=s.texData.get(v.dataId);I(D.isPacked,()=>"batchMatMul result is expected to be packed"),l.shape=N,D.shape=t.outShape,g=ge({inputs:{x:v},backend:s}),g.shape=t.outShape,m.push(v)}else{const w=t.outHeight*t.outWidth,$=S({inputs:{x:n},backend:s,attrs:{shape:f?[t.batchSize,w,t.inChannels]:[t.batchSize,t.inChannels,w]}}),N=S({inputs:{x:e},backend:s,attrs:{shape:[1,t.inChannels,t.outChannels]}}),T=Kn({a:f?$:N,b:f?N:$,transposeA:!f,transposeB:x,backend:s,bias:r,activation:a,preluActivationWeights:o,leakyreluAlpha:i});g=S({inputs:{x:T},backend:s,attrs:{shape:t.outShape}}),m.push($),m.push(N),m.push(T)}for(const w of m)s.disposeIntermediateTensorInfo(w);return g}function lc({x:n,filter:e,convInfo:t,backend:s,bias:r=null,preluActivationWeights:o=null,leakyreluAlpha:i=0,activation:a=null}){const{filterWidth:c,filterHeight:l,inChannels:u,outWidth:d,outHeight:h,dataFormat:f}=t,p=f==="channelsLast",x=c*l*u,g=h*d,m=[t.batchSize,x,g],C=!0,b=!1,w=[];if(o!=null){const re=Yn(o.shape,p);re!=null&&(o=S({inputs:{x:o},backend:s,attrs:{shape:re}}),w.push(o))}if(r!=null){const re=Yn(r.shape,p);re!=null&&(r=S({inputs:{x:r},backend:s,attrs:{shape:re}}),w.push(r))}const $=S({inputs:{x:e},backend:s,attrs:{shape:[1,x,E(e.shape)/x]}});w.push($);const N=new Db(m,t),T=[n.shape,[t.padInfo.top,t.padInfo.left],[t.strideHeight,t.strideWidth],[t.dilationHeight,t.dilationWidth],[t.inChannels],[t.filterWidth*t.inChannels],[t.outWidth]],v=s.runWebGLProgram(N,[n],"float32",T),D=S({inputs:{x:v},backend:s,attrs:{shape:m}});w.push(v),w.push(D);const O=r!=null,L=o!=null,M=a==="leakyrelu",fe=a?pn(a,!0):null,K=new Ja(p?D.shape:$.shape,p?$.shape:D.shape,p?[t.batchSize,g,t.outChannels]:[t.batchSize,t.outChannels,g],C,b,O,fe,L,M),ne=p?[D,$]:[$,D];if(r&&ne.push(r),L&&ne.push(o),M){const re=s.makeTensorInfo([],"float32",Xt(i,"float32"));ne.push(re),w.push(re)}const we=s.runWebGLProgram(K,ne,"float32"),ke=S({inputs:{x:we},backend:s,attrs:{shape:t.outShape}});w.push(we);for(const re of w)s.disposeIntermediateTensorInfo(re);return ke}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ob(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,filter:o}=e,{strides:i,pad:a,dataFormat:c,dilations:l,dimRoundingMode:u}=s,d=Kt(c),h=Be(r.shape,o.shape,i,l,a,u,!1,d);let f;if(h.filterHeight===1&&h.filterWidth===1&&h.dilationHeight===1&&h.dilationWidth===1&&h.strideHeight===1&&h.strideWidth===1&&(h.padInfo.type==="SAME"||h.padInfo.type==="VALID"))f=cc({x:r,filter:o,convInfo:h,backend:t});else if(h.strideWidth<=2&&d==="channelsLast"&&y().getBool("WEBGL_EXP_CONV")){const x=new ac(h),g=[[h.padInfo.top,h.padInfo.left],[h.strideHeight,h.strideWidth],[h.dilationHeight,h.dilationWidth],[h.inHeight,h.inWidth]];f=t.runWebGLProgram(x,[r,o],"float32",g)}else if(y().getBool("WEBGL_CONV_IM2COL"))f=lc({x:r,filter:o,convInfo:h,backend:t});else{const x=new ic(h);f=t.runWebGLProgram(x,[r,o],"float32")}const p=S({inputs:{x:f},backend:t,attrs:{shape:h.outShape}});return t.disposeIntermediateTensorInfo(f),p}const Pb={kernelName:ml,backendName:"webgl",kernelFunc:Ob};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class _b{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,s=e.strideWidth,r=e.padInfo.top,o=e.padInfo.left,i=e.dataFormat==="channelsLast";this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${o};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              ${i?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Lb{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,r=e.strideHeight,o=e.strideWidth,i=e.dataFormat==="channelsLast",a=t-1-e.padInfo.top,c=s-1-e.padInfo.left,l=i?1:2,u=i?2:3,d=i?3:1;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${c});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${d}];

        ivec2 dyCorner = ivec2(coords[${l}], coords[${u}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${o}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            for (int d2 = 0; d2 < ${e.outChannels}; d2++) {

              if (${i}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Bb{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideDepth,s=e.strideHeight,r=e.strideWidth,o=e.padInfo.front,i=e.padInfo.top,a=e.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yF = 0; yF < ${e.outDepth}; yF++) {
            int xF = wF + yF * ${t} - ${o};

            if (xF < 0 || xF >= ${e.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${e.outHeight}; yR++) {
              int xR = wR + yR * ${s} - ${i};

              if (xR < 0 || xR >= ${e.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${e.outWidth}; yC++) {
                int xC = wC + yC * ${r} - ${a};

                if (xC < 0 || xC >= ${e.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Mb{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterDepth,s=e.filterHeight,r=e.filterWidth,o=e.strideDepth,i=e.strideHeight,a=e.strideWidth,c=t-1-e.padInfo.front,l=s-1-e.padInfo.top,u=r-1-e.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${c}, ${l}, ${u});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${t}; wF++) {
          float dyF = float(dyFCorner + wF) / ${o}.0;

          if (dyF < 0.0 || dyF >= ${e.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${t} - 1 - wF;

          for (int wR = 0; wR < ${s}; wR++) {
            float dyR = float(dyRCorner + wR) / ${i}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${s} - 1 - wR;

            for (int wC = 0; wC < ${r}; wC++) {
              float dyC = float(dyCCorner + wC) / ${a}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${r} - 1 - wC;

              for (int d2 = 0; d2 < ${e.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vb(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,dy:o}=e,{strides:i,pad:a,dataFormat:c,dimRoundingMode:l,filterShape:u}=s,d=Kt(c),h=Be(r.shape,u,i,1,a,l,!1,d),f=new _b(h);return t.runWebGLProgram(f,[r,o],"float32")}const Ub={kernelName:gl,backendName:"webgl",kernelFunc:Vb};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Wb{constructor(e){this.variableNames=["dy","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"strides",type:"vec2"}],this.outputShape=e.inShape,this.enableShapeUniforms=se(this.outputShape.length);const t=e.filterHeight,s=e.filterWidth,r=t-1-e.padInfo.top,o=s-1-e.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${r}, ${o});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            int wCPerm = ${s} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${e.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${e.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${e.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gb(n){const{inputs:e,backend:t,attrs:s}=n,{dy:r,filter:o}=e,{inputShape:i,strides:a,pad:c,dataFormat:l,dimRoundingMode:u}=s,d=Kt(l),h=Be(i,o.shape,a,1,c,u,!1,d);if(y().getBool("WEBGL_PACK_CONV2DTRANSPOSE")&&d==="channelsLast"){const f=[[h.strideHeight,h.strideWidth]],p=new Wb(h);return t.runWebGLProgram(p,[r,o],"float32",f)}else{const f=new Lb(h);return t.runWebGLProgram(f,[r,o],"float32")}}const zb={kernelName:xl,backendName:"webgl",kernelFunc:Gb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hb(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,filter:o}=e,{strides:i,pad:a,dilations:c}=s,l=wn(r.shape,o.shape,i,c,a),u=new Fb(l);return t.runWebGLProgram(u,[r,o],"float32")}const Xb={kernelName:Cl,backendName:"webgl",kernelFunc:Hb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jb(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,dy:o}=e,{strides:i,pad:a,filterShape:c}=s,l=wn(r.shape,c,i,1,a),u=new Bb(l);return t.runWebGLProgram(u,[r,o],"float32")}const qb={kernelName:bl,backendName:"webgl",kernelFunc:jb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kb(n){const{inputs:e,backend:t,attrs:s}=n,{dy:r,filter:o}=e,{pad:i,strides:a,inputShape:c}=s,l=wn(c,o.shape,a,1,i),u=new Mb(l);return t.runWebGLProgram(u,[r,o],"float32")}const Yb={kernelName:wl,backendName:"webgl",kernelFunc:Kb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qb=nn+`
  return cos(x);
`,Zb=`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${Et}
  return result;
`,Jb=_({opSnippet:Qb,packedOpSnippet:Zb}),ew={kernelName:yl,backendName:"webgl",kernelFunc:Jb};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tw=`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`,nw=_({opSnippet:tw}),sw={kernelName:$l,backendName:"webgl",kernelFunc:nw};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class rw{constructor(e,t,s,r,o){this.variableNames=["Image","Boxes","BoxInd"],this.outputShape=[];const[i,a,c,l]=e,[u]=t,[d,h]=s;this.outputShape=[u,d,h,l];const f=r==="bilinear"?1:0,[p,x]=[`${a-1}.0`,`${c-1}.0`],[g,m,C]=d>1?[`${(a-1)/(d-1)}`,"(y2-y1) * height_ratio",`y1*${p} + float(y)*(height_scale)`]:["0.0","0.0",`0.5 * (y1+y2) * ${p}`],[b,w,$]=h>1?[`${(c-1)/(h-1)}`,"(x2-x1) * width_ratio",`x1*${x} + float(x)*(width_scale)`]:["0.0","0.0",`0.5 * (x1+x2) * ${x}`];this.userCode=`
      const float height_ratio = float(${g});
      const float width_ratio = float(${b});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${i}) {
          return;
        }

        float height_scale = ${m};
        float width_scale = ${w};

        float in_y = ${C};
        if( in_y < 0.0 || in_y > ${p} ) {
          setOutput(float(${o}));
          return;
        }
        float in_x = ${$};
        if( in_x < 0.0 || in_x > ${x} ) {
          setOutput(float(${o}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${f} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ow=n=>{const{inputs:e,backend:t,attrs:s}=n,{image:r,boxes:o,boxInd:i}=e,{cropSize:a,method:c,extrapolationValue:l}=s,u=new rw(r.shape,o.shape,a,c,l);return t.runWebGLProgram(u,[r,o,i],"float32")},iw={kernelName:Il,backendName:"webgl",kernelFunc:ow};var gn;(function(n){n.Prod="*",n.Sum="+"})(gn||(gn={}));class Co{constructor(e,t,s,r){this.op=e,this.outputShape=t,this.variableNames=["x"],this.customUniforms=[{name:"index",type:"float"}];const o=this.outputShape.length,i=this.op===gn.Prod?"1.0":"0.0",a=s?i:`getX(${bo(o,"coords",this.op)})`,c=this.outputShape[this.outputShape.length-1];let l="",u="";s?(l=r?`end != ${c-1}`:"end != 0",u=r?"end + 1":"end - 1"):(l=r?`end + pow2 < ${c}`:"end >= pow2",u=r?"end + pow2":"end - pow2"),this.userCode=`
      void main() {
        ${U(o)} coords = getOutputCoords();
        int end = ${wo(o,"coords",this.op)};
        float val = ${a};
        int pow2 = int(pow(2.0, index));
        if (${l}) {
          int idx = ${u};
          ${wo(o,"coords",this.op)} = idx;
          val ${this.op}= getX(${bo(o,"coords",this.op)});
        }
        setOutput(val);
      }
    `}}function bo(n,e,t){if(n===1)return`${e}`;if(n===2)return`${e}.x, ${e}.y`;if(n===3)return`${e}.x, ${e}.y, ${e}.z`;if(n===4)return`${e}.x, ${e}.y, ${e}.z, ${e}.w`;throw new Error(`Cumulative ${t} for rank ${n} is not yet supported`)}function wo(n,e,t){if(n===1)return`${e}`;if(n===2)return`${e}.y`;if(n===3)return`${e}.z`;if(n===4)return`${e}.w`;throw new Error(`Cumulative ${t} for rank ${n} is not yet supported`)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uc(n,e,t,s,r,o){const i=e.shape.length,a=Te([s],i);let c=e;a!=null&&(c=ce({inputs:{x:e},backend:t,attrs:{perm:a}}));const l=Ee(1,i)[0];if(l!==i-1)throw new Error(`WebGL cumprod shader expects an inner-most axis=${e.shape.length-1} but got axis=${s}`);const u=c.shape[l];let d=ge({inputs:{x:c},backend:t});for(let h=0;h<=Math.ceil(Math.log2(u))-1;h++){const f=new Co(n,c.shape,!1,o),p=[[h]],x=d;d=t.runWebGLProgram(f,[d],d.dtype,p),t.disposeIntermediateTensorInfo(x)}if(r){const h=new Co(n,c.shape,r,o),f=d;d=t.runWebGLProgram(h,[d],d.dtype),t.disposeIntermediateTensorInfo(f)}if(a!=null){const h=rr(a),f=ce({inputs:{x:d},backend:t,attrs:{perm:h}});return t.disposeIntermediateTensorInfo(d),t.disposeIntermediateTensorInfo(c),f}return d}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aw(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o,exclusive:i,reverse:a}=s;return uc(gn.Prod,r,t,o,i,a)}const cw={kernelName:vl,backendName:"webgl",kernelFunc:aw};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lw(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o,exclusive:i,reverse:a}=s;return uc(gn.Sum,r,t,o,i,a)}const uw={kernelName:Sl,backendName:"webgl",kernelFunc:lw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dw(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,weights:o}=e,{size:i,binaryOutput:a}=s;if(r.shape.length===1){const c=t.readSync(r.dataId),l=t.readSync(o.dataId),u=za(c,l,o.dtype,o.shape,i);return t.makeTensorInfo([i],o.dtype,u)}else if(r.shape.length===2){const c=t.bufferSync(r),l=t.bufferSync(o),u=mx(c,l,i,a);return t.makeTensorInfo(u.shape,o.dtype,u.values)}throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank${r.shape.length}.`)}const hw={kernelName:Rl,backendName:"webgl",kernelFunc:dw};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fw{constructor(e,t,s){this.variableNames=["x"],this.outputShape=[],this.outputShape=e,this.blockSize=t,this.dataFormat=s,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${t};
      int offset_h = imod(h, ${t});
      int in_w = w / ${t};
      int offset_w = imod(w, ${t});
      int offset_d = (offset_h * ${t} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat==="NHWC"?"coords[1]":"coords[2]"}getWidthCoordString(){return this.dataFormat==="NHWC"?"coords[2]":"coords[3]"}getDepthCoordString(){return this.dataFormat==="NHWC"?"coords[3]":"coords[1]"}getOutputDepthSize(){return this.dataFormat==="NHWC"?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat==="NHWC"?"getX(b, in_h, in_w, in_d)":"getX(b, in_d, in_h, in_w)"}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pw(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{blockSize:o,dataFormat:i}=s,a=r.shape[0],c=i==="NHWC"?r.shape[1]:r.shape[2],l=i==="NHWC"?r.shape[2]:r.shape[3],u=i==="NHWC"?r.shape[3]:r.shape[1],d=c*o,h=l*o,f=u/(o*o),p=i==="NHWC"?[a,d,h,f]:[a,f,d,h],x=new fw(p,o,i);return t.runWebGLProgram(x,[r],r.dtype)}const mw={kernelName:Tl,backendName:"webgl",kernelFunc:pw};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class dc{constructor(e,t=!1,s=null,r=!1,o=!1){this.variableNames=["x","W"],this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=se(this.outputShape.length);const i=e.filterHeight,a=e.filterWidth,c=e.outChannels/e.inChannels;let l="",u="";s&&(r?l=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:o?l=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:l=`
          float activation(float x) {
            ${s}
          }
        `,u="result = activation(result);");const d=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),o&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${l}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${c};
        int q = d2 - d1 * ${c};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${i}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${a}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${d}
        ${u}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hc{constructor(e,t=!1,s=null,r=!1,o=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=e.outShape,this.enableShapeUniforms=se(this.outputShape.length);const i=e.outChannels/e.inChannels,a=e.padInfo.left,c=e.strideWidth,l=e.dilationWidth,u=e.filterHeight,d=e.filterWidth,h=d;let f=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let m=0;m<d;m++)f+=`
          vec4 xTexelC${m*2};
          int xTexelC${m*2}Ready;
          vec4 xTexelC${m*2+1};
          int xTexelC${m*2+1}Ready;
          vec4 xC${m};`;f+=`
    for (int r = 0; r < ${u}; r++) {
      `;for(let m=0;m<d;m++)f+=`
          xTexelC${m*2} = vec4(0.0);
          xTexelC${m*2}Ready = 0;
          xTexelC${m*2+1} = vec4(0.0);
          xTexelC${m*2+1}Ready = 0;
          xC${m} = vec4(0.0);`;f+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let m=0;m<(h+1)/2;m++){const C=m*2;if(f+=`
          xC = xCCorner + ${C*l};
          `,c===1){if(C<d&&(a%2===1?(f+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }
              `,l===1&&C>0?f+=`
                xC${C} = vec4(xTexelC${C-2}.zw, xTexelC${C}.xy);
                `:f+=`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${C} = vec4(previous.zw, xTexelC${C}.xy);
                  } else {
                    xC${C} = vec4(0.0, 0.0, xTexelC${C}.xy);
                  }
                  `):f+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }

                xC${C} = xTexelC${C};
                `,C+1<d)){const b=a%2===0?Hs(l):l;l%2===0&&a%2===1||l%2!==0&&a%2!==1?(f+=`
                  xCOffset = xC + imod(pads[1], 2) + ${b};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C+1}Ready == 0) {
                    xTexelC${C+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${C+1}.zw = vec2(0.0);
                    }
                    xTexelC${C+1}Ready = 1;
                  }
                  `,l>1?f+=`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${C+1} = vec4(previous.zw, xTexelC${C+1}.xy);
                    } else {
                     xC${C+1} = vec4(0.0, 0.0, xTexelC${C+1}.xy);
                    }
                    `:f+=`
                    xC${C+1} = vec4(xTexelC${C}.zw, xTexelC${C+1}.xy);
                    `):b===1?f+=`
                    xC${C+1} = xTexelC${C};
                    `:f+=`
                    xCOffset = xC + ${b};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C+1}Ready == 0) {
                      xTexelC${C+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${C+1}.zw = vec2(0.0);
                      }
                      xTexelC${C+1}Ready = 1;
                    }

                    xC${C+1} = xTexelC${C+1};
                    `}}else C<d&&(a%2===1?(f+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${C+1}Ready == 0) {
                  xTexelC${C+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${C+1}.zw = vec2(0.0);
                  }
                  xTexelC${C+1}Ready = 1;
                }

                xC${C} = vec4(xTexelC${C}.zw, xTexelC${C+1}.zw);
              `,C+1<d&&(f+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${C+1} = vec4(xTexelC${C+1}.xy, final.xy);
                `)):(f+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${C}Ready == 0) {
                  xTexelC${C} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${C}.zw = vec2(0.0);
                  }
                  xTexelC${C}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${C+1}Ready == 0) {
                  xTexelC${C+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${C+1}.zw = vec2(0.);
                  }
                  xTexelC${C+1}Ready = 1;
                }

                xC${C} = vec4(
                  xTexelC${C}.xy, xTexelC${C+1}.xy);
              `,C+1<d&&(f+=`
                  xC${C+1} = vec4(xTexelC${C}.zw, xTexelC${C+1}.zw);
                `)));C<d&&(f+=`
            wTexel = getW(r, ${C}, d1, q);
            dotProd += xC${C} * vec4(wTexel.xz, wTexel.xz);
          `,C+1<d&&(f+=`
              wTexel = getW(r, ${C+1}, d1, q);
              dotProd += xC${C+1} * vec4(wTexel.xz, wTexel.xz);
            `))}f+=`
    }
  `,f+=`
      }
    `;let p="",x="";s&&(r?p=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:o?p=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:p=`vec4 activation(vec4 x) {
          ${s}
        }`,x="result = activation(result);");const g=t?"result += getBiasAtOutCoords();":"";t&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),o&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${p}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${i};
        int q = d2 - d1 * ${i};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${f}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${g}
        ${x}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gw(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,filter:o}=e,{strides:i,pad:a,dilations:c,dimRoundingMode:l}=s;let u=c;u==null&&(u=[1,1]),I(qt(i,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${i} and dilations '${u}'`);const d=Be(r.shape,o.shape,i,u,a,l,!0);let h;y().getBool("WEBGL_PACK_DEPTHWISECONV")&&d.strideWidth<=2&&d.outChannels/d.inChannels===1?h=new hc(d):h=new dc(d);const f=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];return t.runWebGLProgram(h,[r,o],"float32",f)}const xw={kernelName:El,backendName:"webgl",kernelFunc:gw};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Cw{constructor(e){this.variableNames=["x","dy"],this.outputShape=e.filterShape;const t=e.strideHeight,s=e.strideWidth,r=e.padInfo.top,o=e.padInfo.left,i=e.outChannels/e.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${i} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${e.batchSize}; b++) {
          for (int yR = 0; yR < ${e.outHeight}; yR++) {
            int xR = wR + yR * ${t} - ${r};

            if (xR < 0 || xR >= ${e.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${e.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${o};

              if (xC < 0 || xC >= ${e.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class bw{constructor(e){this.variableNames=["dy","W"],this.outputShape=e.inShape;const t=e.filterHeight,s=e.filterWidth,r=e.strideHeight,o=e.strideWidth,i=t-1-e.padInfo.top,a=s-1-e.padInfo.left,c=e.outChannels/e.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${i}, ${a});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${t}; wR++) {
          float dyR = float(dyRCorner + wR) / ${r}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${t} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${o}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${c}; dm++) {
              int d2 = d1 * ${c} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ww(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,dy:o}=e,{strides:i,dilations:a,pad:c,dimRoundingMode:l,filterShape:u}=s,d=Be(r.shape,u,i,a,c,l,!0),h=new Cw(d);return t.runWebGLProgram(h,[r,o],"float32")}const yw={kernelName:Nl,backendName:"webgl",kernelFunc:ww};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $w(n){const{inputs:e,backend:t,attrs:s}=n,{dy:r,filter:o}=e,{strides:i,dilations:a,pad:c,dimRoundingMode:l,inputShape:u}=s,d=Be(u,o.shape,i,a,c,l,!0),h=new bw(d);return t.runWebGLProgram(h,[r,o],"float32")}const vw={kernelName:kl,backendName:"webgl",kernelFunc:$w};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Sw{constructor(e){this.variableNames=["X"],this.outputShape=[e,e],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Iw(n){const{inputs:e,backend:t}=n,{x:s}=e,r=[...s.shape,...s.shape],o=E(s.shape),i=S({inputs:{x:s},backend:t,attrs:{shape:[o]}}),a=new Sw(o),c=t.runWebGLProgram(a,[i],i.dtype),l=S({inputs:{x:c},backend:t,attrs:{shape:r}});return t.disposeIntermediateTensorInfo(i),t.disposeIntermediateTensorInfo(c),l}const Rw={kernelName:Al,backendName:"webgl",kernelFunc:Iw};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Tw{constructor(e){this.variableNames=["x","W"],this.outputShape=e.outShape;const{inHeight:t,inWidth:s,padInfo:r,strideHeight:o,strideWidth:i,filterHeight:a,filterWidth:c,dilationHeight:l,dilationWidth:u}=e,{top:d,left:h}=r;this.userCode=`
      const ivec2 strides = ivec2(${o}, ${i});
      const ivec2 pads = ivec2(${d}, ${h});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${a}; h++) {
          int hIn = hBeg + h * ${l};

          if (hIn >= 0 && hIn < ${t}) {
            for (int w = 0; w < ${c}; w++) {
              int wIn = wBeg + w * ${u};

              if (wIn >= 0 && wIn < ${s}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ew(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,filter:o}=e,{strides:i,pad:a,dilations:c}=s,l=Ni(r.shape,o.shape,i,a,"NHWC",c);let u;const d=new Tw(l);u=t.runWebGLProgram(d,[r,o],"float32");const h=S({inputs:{x:u},backend:t,attrs:{shape:l.outShape}});return t.disposeIntermediateTensorInfo(u),h}const Nw={kernelName:Fl,backendName:"webgl",kernelFunc:Ew};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kw(n){const{inputs:e,backend:t,attrs:s}=n,{equation:r}=s,o=e,{allDims:i,summedDims:a,idDims:c}=ua(r,o.length);ha(i.length,c,o);const{path:l,steps:u}=fa(a,c),d=u.length;let h=null,f=i.length;const p=[];for(let x=0;x<d;++x){for(const g of u[x]){const{permutationIndices:m,expandDims:C}=da(f,c[g]);let b;pa(m)?b=o[g]:(b=ce({inputs:{x:o[g]},backend:t,attrs:{perm:m}}),p.push(b));const w=b.shape.slice();for(let $=0;$<C.length;++$)w.splice(C[$],0,1);J(b.shape,w)||(b=S({inputs:{x:b},backend:t,attrs:{shape:w}}),p.push(b)),h===null?h=b:(h=br({inputs:{a:b,b:h},backend:t}),p.push(h))}x<d-1&&(l[x]>=0&&(h=as({inputs:{x:h},backend:t,attrs:{axis:l[x]-(i.length-f),keepDims:!1}}),p.push(h)),f--)}for(const x of p)x!==h&&t.disposeIntermediateTensorInfo(x);return h}const Aw={kernelName:Dl,backendName:"webgl",kernelFunc:kw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fw="return (x >= 0.0) ? x : (exp(x) - 1.0);",Dw=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,Ow=_({opSnippet:Fw,packedOpSnippet:Dw}),Pw={kernelName:Lo,backendName:"webgl",kernelFunc:Ow};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _w="return (b >= 0.0) ? a : a * (b + 1.0);",Lw=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,Bw=n=>{const{inputs:e,backend:t}=n,{dy:s,y:r}=e,o=y().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new tn(Lw,s.shape,r.shape):new wt(_w,s.shape,r.shape);return t.runWebGLProgram(o,[s,r],s.dtype)},Mw={kernelName:Ol,backendName:"webgl",kernelFunc:Bw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vw=`
  return vec4(equal(a, b));
`,Uw="return float(a == b);",Ww=te({opSnippet:Uw,packedOpSnippet:Vw,dtype:"bool",cpuKernelImpl:wx}),Gw={kernelName:_l,backendName:"webgl",kernelFunc:Ww};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zw=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${ra};
  float a1 = ${oa};
  float a2 = ${ia};
  float a3 = ${aa};
  float a4 = ${ca};
  float a5 = ${la};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`,Hw=_({opSnippet:zw}),Xw={kernelName:Pl,backendName:"webgl",kernelFunc:Hw};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jw=nn+`
  return exp(x);
`,qw=`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,fc=_({opSnippet:jw,packedOpSnippet:qw,cpuKernelImpl:yx,dtype:"float32"}),Kw={kernelName:Ll,backendName:"webgl",kernelFunc:fc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zs(n){const{inputs:e,attrs:t,backend:s}=n,{dim:r}=t,{input:o}=e,i=o.shape.length,a=o.shape.slice();let c=r;return r<0&&(I(-(i+1)<=r,()=>`Axis must be in the interval [${-(i+1)}, ${i}]`),c=i+r+1),a.splice(c,0,1),S({inputs:{x:o},backend:s,attrs:{shape:a}})}const Yw={kernelName:Bl,backendName:"webgl",kernelFunc:zs};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yo="return exp(x) - 1.0;",Qw=_({opSnippet:yo,packedOpSnippet:yo,cpuKernelImpl:$x}),Zw={kernelName:Ml,backendName:"webgl",kernelFunc:Qw};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $o{constructor(e,t,s){this.variableNames=["real","imag"];const r=t[1];this.outputShape=t;const o=s?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,i=s?`${r}.0`:"1.0";let a;if(e==="real")a="return real * expR - imag * expI;";else if(e==="imag")a="return real * expI + imag * expR;";else throw new Error(`FFT component must be either "real" or "imag", got ${e}.`);this.userCode=`
      const float exponentMultiplier = ${o};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${a}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${r});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${r}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${i};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pc(n,e,t){const s=t.texData.get(n.dataId),r=E(n.shape),o=n.shape[n.shape.length-1],i=r/o,a=S({inputs:{x:n},backend:t,attrs:{shape:[i,o]}}),c=a.shape,l=new $o("real",c,e),u=new $o("imag",c,e),d=[{dataId:s.complexTensorInfos.real.dataId,dtype:s.complexTensorInfos.real.dtype,shape:c},{dataId:s.complexTensorInfos.imag.dataId,dtype:s.complexTensorInfos.imag.dtype,shape:c}],h=t.runWebGLProgram(l,d,"float32"),f=t.runWebGLProgram(u,d,"float32"),p=rt({inputs:{real:h,imag:f},backend:t});t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f);const x=S({inputs:{x:p},backend:t,attrs:{shape:n.shape}});return t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(p),x}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jw(n){const{inputs:e,backend:t}=n,{input:s}=e;return pc(s,!1,t)}const e1={kernelName:Vl,backendName:"webgl",kernelFunc:Jw};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class t1{constructor(e,t){this.outputShape=[],this.customUniforms=[{name:"value",type:"float"}],this.variableNames=["x"],this.outputShape=e,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function In(n){const{backend:e,attrs:t}=n,{shape:s,value:r}=t;let{dtype:o}=t;if(o=o||xn(r),o==="string"){const i=q(o,E(s));return i.fill(r),e.makeTensorInfo(s,o,i)}else{const i=new t1(s,r),a=[[r]];return e.runWebGLProgram(i,[],o,a)}}const n1={kernelName:Bo,backendName:"webgl",kernelFunc:In};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class s1{constructor(e){this.variableNames=["Image"],this.outputShape=[];const t=e[2];this.outputShape=e,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${t} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${t}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const r1={kernelName:Ul,backendName:"webgl",kernelFunc:({inputs:n,backend:e})=>{const{image:t}=n,s=e,r=new s1(t.shape);return s.runWebGLProgram(r,[t],t.dtype)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vo="return floor(x);",o1=_({opSnippet:vo,packedOpSnippet:vo,cpuKernelImpl:vx}),i1={kernelName:Wl,backendName:"webgl",kernelFunc:o1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const a1=`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,c1=`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,l1=te({opSnippet:a1,packedOpSnippet:c1,dtype:"int32"}),u1={kernelName:Mo,backendName:"webgl",kernelFunc:l1};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class d1{constructor(e){this.variableNames=["A"];const t=le(),[s,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${r}.0, ${s}.0);

        vec4 values = ${t.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class h1{constructor(e){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const t=le(),[s,r]=e;this.outputShape=e,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${r}.0, ${s}.0);
            vec4 values = ${t.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${t.output} = result;
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const f1={kernelName:vd,backendName:"webgl",kernelFunc:p1};let Ft,ys=y().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");function p1(n){const{inputs:e,backend:t,attrs:s}=n;let{pixels:r}=e;const{numChannels:o}=s,i=typeof HTMLVideoElement<"u"&&r instanceof HTMLVideoElement,a=typeof HTMLImageElement<"u"&&r instanceof HTMLImageElement,[c,l]=i?[r.videoWidth,r.videoHeight]:[r.width,r.height],u=[l,c],d=[l,c,o];if(a||i){const x=y().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");(Ft==null||x!==ys)&&(ys=x,Ft=document.createElement("canvas").getContext("2d",{willReadFrequently:ys})),Ft.canvas.width=c,Ft.canvas.height=l,Ft.drawImage(r,0,0,c,l),r=Ft.canvas}const h=t.makeTensorInfo(u,"int32");t.texData.get(h.dataId).usage=Ce.PIXELS,t.gpgpu.uploadPixelDataToTexture(t.getTexture(h.dataId),r);const f=y().getBool("WEBGL_PACK")?new h1(d):new d1(d),p=t.runWebGLProgram(f,[h],"int32");return t.disposeData(h.dataId),p}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function m1(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,filter:o,bias:i,preluActivationWeights:a}=e,{strides:c,pad:l,dataFormat:u,dilations:d,dimRoundingMode:h,activation:f,leakyreluAlpha:p}=s,x=Kt(u),g=Be(r.shape,o.shape,c,d,l,h,!1,x);let m;const C=[],b=i!=null,w=a!=null,$=f==="leakyrelu",N=()=>{const v=[r,o],D=(O,L)=>{if(L==="NCHW"&&O.shape.length===1&&O.shape[0]!==1){const M=S({inputs:{x:O},backend:t,attrs:{shape:[O.shape[0],1,1]}});return C.push(M),M}return O};if(b&&v.push(D(i,u)),w&&v.push(D(a,u)),$){const O=t.makeTensorInfo([],"float32",Xt(p,"float32"));v.push(O),C.push(O)}return v};if(g.filterHeight===1&&g.filterWidth===1&&g.dilationHeight===1&&g.dilationWidth===1&&g.strideHeight===1&&g.strideWidth===1&&(g.padInfo.type==="SAME"||g.padInfo.type==="VALID"))m=cc({x:r,filter:o,convInfo:g,backend:t,bias:i,activation:f,preluActivationWeights:a,leakyreluAlpha:p});else if(g.strideWidth<=2&&x==="channelsLast"&&y().getBool("WEBGL_EXP_CONV")){const v=f?pn(f,!0):null,D=new ac(g,b,v,w,$),O=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],L=N();m=t.runWebGLProgram(D,L,"float32",O)}else if(y().getBool("WEBGL_CONV_IM2COL"))m=lc({x:r,filter:o,convInfo:g,backend:t,bias:i,activation:f,preluActivationWeights:a,leakyreluAlpha:p});else{const v=f?pn(f,!1):null,D=new ic(g,b,v,w,$),O=N();m=t.runWebGLProgram(D,O,"float32")}const T=S({inputs:{x:m},backend:t,attrs:{shape:g.outShape}});return C.push(m),C.forEach(v=>t.disposeIntermediateTensorInfo(v)),T}const g1={kernelName:Rd,backendName:"webgl",kernelFunc:m1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function x1(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,filter:o,bias:i,preluActivationWeights:a}=e,{strides:c,pad:l,dilations:u,dimRoundingMode:d,activation:h,leakyreluAlpha:f}=s,p=[];let x=u;x==null&&(x=[1,1]),I(qt(c,x),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${c} and dilations '${x}'`);const g=Be(r.shape,o.shape,c,x,l,d,!0),m=y().getBool("WEBGL_PACK_DEPTHWISECONV")&&g.strideWidth<=2&&g.outChannels/g.inChannels===1,C=h?pn(h,m):null,b=[r,o],w=i!=null,$=a!=null,N=h==="leakyrelu";if(w&&b.push(i),$&&b.push(a),N){const O=t.makeTensorInfo([],"float32",Xt(f,"float32"));b.push(O),p.push(O)}let T;m?T=new hc(g,w,C,$,N):T=new dc(g,w,C,$,N);const v=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],D=t.runWebGLProgram(T,b,"float32",v);return p.forEach(O=>t.disposeIntermediateTensorInfo(O)),D}const C1={kernelName:Td,backendName:"webgl",kernelFunc:x1};class b1{constructor(e,t,s,r){this.sliceDim=e,this.strides=t,this.paramsShape=r,this.variableNames=["x","indices"],this.outputShape=s;const o=U(s.length);let i=`
    int index;`;for(let a=0;a<this.sliceDim;a++)i+=`
          index = round(getIndices(coords[0], ${a}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[a]};
          flattenIndex += index * ${this.strides[a]};`;this.userCode=`
         void main() {
          ${o} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${i}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function w1(n){const{inputs:e,backend:t}=n,{params:s,indices:r}=e,o=r.shape,i=o[o.length-1],a=E(s.shape),[c,l,u,d]=Pi(s,r),h=S({inputs:{x:r},backend:t,attrs:{shape:[l,i]}}),f=S({inputs:{x:s},backend:t,attrs:{shape:[E(s.shape)/u,u]}});if(t.shouldExecuteOnCPU([s,r])||s.dtype==="string"){const m=t.readSync(r.dataId),C=t.bufferSync(s),b=Sx(m,C,s.dtype,l,i,u,d,s.shape,a);return t.makeTensorInfo(c,s.dtype,b.values)}const p=new b1(i,d,[l,u],s.shape),x=t.runWebGLProgram(p,[f,h],f.dtype),g=S({inputs:{x},backend:t,attrs:{shape:c}});return t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(x),g}const y1={kernelName:Hl,backendName:"webgl",kernelFunc:w1};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $1{constructor(e,t){this.variableNames=["A","indices"],this.outputShape=t,this.rank=t.length;const s=U(this.rank),r=v1(e);this.userCode=`
      void main() {
        ${s} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${e[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${r}));
      }
    `}}function v1(n,e){const t=["resRC.x","resRC.y","resRC.z","resRC.w"],s=[];for(let r=0;r<n.length;r++)r===2?s.push("index"):s.push(`${t[r]}`);return s.join()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mc(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,indices:o}=e,{axis:i,batchDims:a}=s,c=de(i,r.shape)[0];if(y().get("DEBUG")){const C=t.readSync(o.dataId),b=r.shape[c];for(let w=0;w<C.length;++w){const $=C[w];I($<=b-1&&$>=0,()=>`GatherV2: the index value ${$} is not in [0, ${b-1}]`)}}const l=Na(r,o,c,a),u=E(o.shape),d=[],h=S({inputs:{x:r},backend:t,attrs:{shape:[l.batchSize,l.outerSize,l.dimSize,l.sliceSize]}}),f=S({inputs:{x:o},backend:t,attrs:{shape:[l.batchSize,u/l.batchSize]}});d.push(h),d.push(f);const p=[l.batchSize,l.outerSize,u/l.batchSize,l.sliceSize];if(t.shouldExecuteOnCPU([r,o])||r.dtype==="string"){const C=t.bufferSync(f),b=t.bufferSync(h),w=Ix(b,C,p);return d.forEach($=>t.disposeIntermediateTensorInfo($)),t.makeTensorInfo(l.outputShape,w.dtype,w.values)}const x=new $1(h.shape,p),g=t.runWebGLProgram(x,[h,f],h.dtype);d.push(g);const m=S({inputs:{x:g},backend:t,attrs:{shape:l.outputShape}});return d.forEach(C=>t.disposeIntermediateTensorInfo(C)),m}const S1={kernelName:zl,backendName:"webgl",kernelFunc:mc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const I1="return float(a > b);",R1=`
  return vec4(greaterThan(a, b));
`,T1=te({opSnippet:I1,packedOpSnippet:R1,cpuKernelImpl:Rx,dtype:"bool"}),E1={kernelName:Xl,backendName:"webgl",kernelFunc:T1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const N1="return float(a >= b);",k1=`
  return vec4(greaterThanEqual(a, b));
`,A1=te({opSnippet:N1,packedOpSnippet:k1,dtype:"bool",cpuKernelImpl:Tx}),F1={kernelName:jl,backendName:"webgl",kernelFunc:A1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D1(n){const{inputs:e,backend:t}=n,{input:s}=e;return pc(s,!0,t)}const O1={kernelName:ql,backendName:"webgl",kernelFunc:D1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const P1="return float(!isnan(x) && !isinf(x));",_1=_({opSnippet:P1,dtype:"bool"}),L1={kernelName:Yl,backendName:"webgl",kernelFunc:_1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const B1="return float(isinf(x));",M1=_({opSnippet:B1,dtype:"bool"}),V1={kernelName:Ql,backendName:"webgl",kernelFunc:M1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const U1="return float(isnan(x));",W1=_({opSnippet:U1,dtype:"bool"}),G1={kernelName:Zl,backendName:"webgl",kernelFunc:W1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const z1="return float(a < b);",H1=`
  return vec4(lessThan(a, b));
`,X1=te({opSnippet:z1,packedOpSnippet:H1,cpuKernelImpl:Ex,dtype:"bool"}),j1={kernelName:Jl,backendName:"webgl",kernelFunc:X1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const q1="return float(a <= b);",K1=`
  return vec4(lessThanEqual(a, b));
`,Y1=te({opSnippet:q1,packedOpSnippet:K1,cpuKernelImpl:Nx,dtype:"bool"}),Q1={kernelName:eu,backendName:"webgl",kernelFunc:Y1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Z1(n){const{backend:e,attrs:t}=n,{start:s,stop:r,num:o}=t,i=kx(s,r,o);return e.makeTensorInfo([i.length],"float32",i)}const J1={kernelName:tu,backendName:"webgl",kernelFunc:Z1};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ey=nn+`
  return x < 0.0 ? 0./0. : log(x);
`,ty=`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,ny=_({opSnippet:ey,packedOpSnippet:ty,cpuKernelImpl:Ax}),sy={kernelName:nu,backendName:"webgl",kernelFunc:ny};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ry=nn+`
  return log(1.0 + x);
`,oy=_({opSnippet:ry}),iy={kernelName:su,backendName:"webgl",kernelFunc:oy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ay="return float(a >= 1.0 && b >= 1.0);",cy=`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,ly=te({opSnippet:ay,packedOpSnippet:cy,dtype:"bool"}),uy={kernelName:ru,backendName:"webgl",kernelFunc:ly};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dy="return float(!(x >= 1.0));",hy=_({opSnippet:dy}),fy={kernelName:ou,backendName:"webgl",kernelFunc:hy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const py="return float(a >= 1.0 || b >= 1.0);",my=`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,gy=te({opSnippet:py,packedOpSnippet:my,dtype:"bool"}),xy={kernelName:iu,backendName:"webgl",kernelFunc:gy};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Cy{constructor(e,t,s,r,o){this.variableNames=["x"],this.outputShape=[];const i=t,a=e[3]-1;this.outputShape=e;let c;const l=`float(${s}) + float(${r}) * sum`;o===.5?c=`inversesqrt(${l})`:o===1?c=`1.0/(${l})`:c=`exp(log(${l}) * float(-${o}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${i}; j <= ${i}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${a}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${c};
        setOutput(val);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class by{constructor(e,t,s,r,o){this.variableNames=["x"],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;const i=t,a=e[3]-1;this.outputShape=e;let c;const l=`float(${s}) + float(${r}) * sum`;o===.5?c=`inversesqrt(${l})`:o===1?c=`1.0/(${l})`:c=`exp(log(${l}) * float(-${o}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${i};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${i}; j <= ${i}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${a}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${c};
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wy=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{depthRadius:o,bias:i,alpha:a,beta:c}=s,l=y().getBool("WEBGL_PACK_NORMALIZATION")?new by(r.shape,o,i,a,c):new Cy(r.shape,o,i,a,c);return t.runWebGLProgram(l,[r],r.dtype)},yy={kernelName:au,backendName:"webgl",kernelFunc:wy};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $y{constructor(e,t,s,r,o){this.variableNames=["inputImage","outputImage","dy"],this.outputShape=[],this.outputShape=e,this.depth=e[3],this.depthRadius=t,this.bias=s,this.alpha=r,this.beta=o,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${t})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${t} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${r}) * norm + float(${s});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${r})
                * float(${o})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${o});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vy=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:r,y:o,dy:i}=e,{depthRadius:a,bias:c,alpha:l,beta:u}=s,d=new $y(r.shape,a,c,l,u);return t.runWebGLProgram(d,[r,o,i],r.dtype)},Sy={kernelName:cu,backendName:"webgl",kernelFunc:vy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Iy(n,e,t,s){const r=E(e),i=E(n.shape)/r,a=S({inputs:{x:n},attrs:{shape:[i,r]},backend:s}),c=Nt(a,n.dtype,"max",s),l=S({inputs:{x:c},attrs:{shape:t},backend:s});return s.disposeIntermediateTensorInfo(a),s.disposeIntermediateTensorInfo(c),l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gc(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{reductionIndices:o,keepDims:i}=s,a=r.shape.length,c=de(o,r.shape);let l=c;const u=Te(l,a),d=u!=null,h=t.shouldExecuteOnCPU([r]);let f=r;if(d){if(h){const b=t.texData.get(f.dataId).values,w=new Array(a);for(let T=0;T<w.length;T++)w[T]=r.shape[u[T]];const $=xr(b,r.shape,r.dtype,u,w);f=t.makeTensorInfo(w,r.dtype);const N=t.texData.get(f.dataId);N.values=$}else f=is(r,u,t);l=Ee(l.length,a)}Me("max",l,a);const[p,x]=He(f.shape,l);let g=p;i&&(g=je(p,c));let m;if(h){const b=t.texData.get(f.dataId).values,w=Fx(b,E(x),g,r.dtype);m=t.makeTensorInfo(g,r.dtype);const $=t.texData.get(m.dataId);$.values=w}else m=Iy(f,x,g,t);return d&&t.disposeIntermediateTensorInfo(f),m}const Ry={kernelName:lu,backendName:"webgl",kernelFunc:gc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ty=Cr+`
  return max(a, b);
`,Ey=`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Et+`
  return result;
`,Ny=te({opSnippet:Ty,packedOpSnippet:Ey,cpuKernelImpl:Dx}),ky={kernelName:Uo,backendName:"webgl",kernelFunc:Ny};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ay(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e;$n(r,"maxPool");const{filterSize:o,strides:i,pad:a,dimRoundingMode:c}=s,l=1;I(qt(i,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${l}'`);const u=jt(r.shape,o,i,l,a,c);if(u.filterWidth===1&&u.filterHeight===1&&J(u.inShape,u.outShape))return ge({inputs:{x:r},backend:t});const d=new mn(u,"max",!1);return t.runWebGLProgram(d,[r],r.dtype)}const Fy={kernelName:uu,backendName:"webgl",kernelFunc:Ay};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dy(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{filterSize:o,strides:i,pad:a,dataFormat:c,dimRoundingMode:l}=s,u=[1,1,1],d=bn(r.shape,o,i,u,a,l,c),h=new wr(d,"max",!1);return t.runWebGLProgram(h,[r],r.dtype)}const Oy={kernelName:hu,backendName:"webgl",kernelFunc:Dy};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Py{constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideHeight,s=e.strideWidth,r=e.dilationHeight,o=e.effectiveFilterHeight,i=e.effectiveFilterWidth,a=o-1-e.padInfo.top,c=i-1-e.padInfo.left,l=o*i-1;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${c});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${o};
          wR += ${r}) {
          float dyR = float(dyRCorner + wR) / ${t}.0;

          if (dyR < 0.0 || dyR >= ${e.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${i}; wC++) {
            float dyC = float(dyCCorner + wC) / ${s}.0;

            if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${l} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${i} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}}class _y{constructor(e){this.variableNames=["dy","maxPos"],this.outputShape=e.inShape;const t=e.strideDepth,s=e.strideHeight,r=e.strideWidth,o=e.dilationDepth,i=e.dilationHeight,a=e.dilationWidth,c=e.effectiveFilterDepth,l=e.effectiveFilterHeight,u=e.effectiveFilterWidth,d=c-1-e.padInfo.front,h=l-1-e.padInfo.top,f=u-1-e.padInfo.left,p=c*l*u-1;this.userCode=`
      const ivec3 pads = ivec3(${d}, ${h}, ${f});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${c};
           wD += ${o}) {
          float dyD = float(dyDCorner + wD) / ${t}.0;

          if (dyD < 0.0 || dyD >= ${e.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${l};
              wR += ${i}) {
            float dyR = float(dyRCorner + wR) / ${s}.0;

            if (dyR < 0.0 || dyR >= ${e.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${u};
                wC += ${a}) {
              float dyC = float(dyCCorner + wC) / ${r}.0;

              if (dyC < 0.0 || dyC >= ${e.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${p} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${l} * ${u} +
                  wR * ${u} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ly(n){const{inputs:e,backend:t,attrs:s}=n,{dy:r,input:o}=e,i=o,{filterSize:a,strides:c,pad:l,dimRoundingMode:u}=s,d=[1,1,1],h=bn(i.shape,a,c,d,l,u),f=new wr(h,"max",!0),p=t.runWebGLProgram(f,[i],i.dtype),x=new _y(h),g=t.runWebGLProgram(x,[r,p],i.dtype);return t.disposeIntermediateTensorInfo(p),g}const By={kernelName:fu,backendName:"webgl",kernelFunc:Ly};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function My(n){const{inputs:e,backend:t,attrs:s}=n,{dy:r,input:o,output:i}=e,a=o;$n([o,i],"maxPoolGrad");const{filterSize:c,strides:l,pad:u,dimRoundingMode:d}=s,h=jt(a.shape,c,l,1,u,d),f=!0,p=new mn(h,"max",f),x=t.runWebGLProgram(p,[a],a.dtype),g=new Py(h),m=t.runWebGLProgram(g,[r,x],a.dtype);return t.disposeIntermediateTensorInfo(x),m}const Vy={kernelName:du,backendName:"webgl",kernelFunc:My};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uy(n,e,t,s){let r=new mn(t,"max",!1);const o=s.runWebGLProgram(r,[n],"float32");r=new mn(t,"max",!0,!0,e);const i=s.runWebGLProgram(r,[n],"float32");return[o,i]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wy={kernelName:pu,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{x:s}=n,{filterSize:r,strides:o,pad:i,includeBatchInIndex:a}=e,c=t;I(s.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${s.shape.length}.`);const l=[1,1];I(qt(o,l),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${o} and dilations '${l}'`);const u=jt(s.shape,r,o,l,i),[d,h]=Uy(s,a,u,c);return[d,h]}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gy(n,e,t,s){const r=E(e),i=E(n.shape)/r,a=S({inputs:{x:n},attrs:{shape:[i,r]},backend:s}),c=Nt(a,"float32","mean",s),l=S({inputs:{x:c},attrs:{shape:t},backend:s});return s.disposeIntermediateTensorInfo(a),s.disposeIntermediateTensorInfo(c),l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zy={kernelName:mu,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{x:s}=n,{keepDims:r,axis:o}=e,i=t,a=s.shape.length,c=de(o,s.shape);let l=c;const u=Te(l,a),d=u!=null,h=i.shouldExecuteOnCPU([s]),f=[];let p=s;if(d){if(h){const w=i.texData.get(p.dataId).values,$=new Array(a);for(let v=0;v<$.length;v++)$[v]=s.shape[u[v]];const N=xr(w,s.shape,s.dtype,u,$);p=i.makeTensorInfo($,s.dtype);const T=i.texData.get(p.dataId);T.values=N}else p=is(s,u,i);f.push(p),l=Ee(l.length,a)}Me("sum",l,a);const[x,g]=He(p.shape,l);let m=x;r&&(m=je(x,c));const C=Gy(p,g,m,i);for(const b of f)i.disposeIntermediateTensorInfo(b);return C}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hy(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o,keepDims:i}=s,a=r.shape.length,c=de(o,r.shape);let l=c;const u=Te(l,a);let d=r;u!=null&&(d=ce({inputs:{x:r},backend:t,attrs:{perm:u}}),l=Ee(l.length,r.shape.length)),Me("min",l,a);const[h,f]=He(d.shape,l),p=E(f),x=S({inputs:{x:d},backend:t,attrs:{shape:[-1,p]}}),g=Nt(x,x.dtype,"min",t);let m;if(i){const C=je(h,c);m=S({inputs:{x:g},backend:t,attrs:{shape:C}})}else m=S({inputs:{x:g},backend:t,attrs:{shape:h}});return t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(g),u!=null&&t.disposeIntermediateTensorInfo(d),m}const Xy={kernelName:gu,backendName:"webgl",kernelFunc:Hy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jy=Cr+`
  return min(a, b);
`,qy=`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Et+`
  return result;
`,Ky=te({opSnippet:jy,packedOpSnippet:qy,cpuKernelImpl:Ox}),Yy={kernelName:xu,backendName:"webgl",kernelFunc:Ky};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Qy{constructor(e,t,s){this.variableNames=["x"],this.outputShape=t.map((u,d)=>u[0]+e[d]+u[1]);const r=e.length,o=U(r),i=t.map(u=>u[0]).join(","),a=t.map((u,d)=>u[0]+e[d]).join(","),c=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,r),l=s==="reflect"?0:1;if(r===1){this.userCode=`
        int start = ${i};
        int end = ${a};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${l};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${l};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${o} start = ${o}(${i});
      ${o} end = ${o}(${a});

      void main() {
        ${o} outC = getOutputCoords();
        for (int i = 0; i < ${r}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${l};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${l};
          }
        }
        ${o} coords = outC - start;
        setOutput(getX(${c}));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Zy{constructor(e,t,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t.map((p,x)=>p[0]+e[x]+p[1]);const r=e.length,o=U(r),i=t.map(p=>p[0]).join(","),a=t.map((p,x)=>p[0]+e[x]).join(","),c=ie("rc",r),l=ie("source",r),u=`${c[r-1]} < ${this.outputShape[r-1]}`,d=r===1?"source":`vec2(${l.slice(-2).join()})`,h=s==="reflect"?0:1;let f="";if(r===1){const p=`
        ${o} source = rc;
        if (source < start) {
          source = start * 2 - source - ${h};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${h};
        }
        source -= start;
      `;f=`
        ${o} rc = outputLoc;
        ${p}
        result[0] = getChannel(getX(${l.join()}), ${d});
        ${c[r-1]} += 1;
        if(${u}) {
          ${p}
          result[1] = getChannel(getX(${l.join()}), ${d});
        }
      `}else{const p=`
        ${o} source = rc;
        ${o} lt = ${o}(lessThan(source, start));
        ${o} gte = ${o}(greaterThanEqual(source, end));
        ${o} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${h}) +
                gte * ((end - 1) * 2 - source + ${h});
        source -= start;
      `;f=`
        ${o} rc = outputLoc;
        ${p}
        result[0] = getChannel(getX(${l.join()}), ${d});
        ${c[r-1]} += 1;
        if(${u}) {
          ${p}
          result[1] = getChannel(getX(${l.join()}), ${d});
        }
        rc = outputLoc;
        ${c[r-2]} += 1;
        if(${c[r-2]} < ${this.outputShape[r-2]}) {
          ${p}
          result[2] = getChannel(getX(${l.join()}), ${d});
          ${c[r-1]} += 1;
          if(${u}) {
            ${p}
            result[3] = getChannel(getX(${l.join()}), ${d});
          }
        }
      `}this.userCode=`
      const ${o} start = ${o}(${i});
      const ${o} end = ${o}(${a});

      void main() {
        ${o} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jy=({inputs:n,backend:e,attrs:t})=>{const{x:s}=n,{paddings:r,mode:o}=t,i=y().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new Zy(s.shape,r,o):new Qy(s.shape,r,o);return e.runWebGLProgram(i,[s],s.dtype)},e$={kernelName:Cu,backendName:"webgl",kernelFunc:Jy};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const t$=`if (b == 0.0) return NAN;
  return mod(a, b);`,n$=`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+Et+`
  return result;
`,s$=te({opSnippet:t$,packedOpSnippet:n$}),r$={kernelName:bu,backendName:"webgl",kernelFunc:s$};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class o${constructor(e,t,s){this.variableNames=["probs"],this.customUniforms=[{name:"seed",type:"float"}],this.outputShape=[e,s],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${t-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${t-1}));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const i$=`
if (a == b) {
  return 1.0;
};
return a / b;`,a$=`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,xc=te({opSnippet:i$,packedOpSnippet:a$,checkOutOfBounds:!0}),c$={kernelName:_o,backendName:"webgl",kernelFunc:xc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const So="return a - b;",Cc=te({opSnippet:So,packedOpSnippet:So,supportsComplex:!0,cpuKernelImpl:t0}),l$={kernelName:Qo,backendName:"webgl",kernelFunc:Cc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bc(n){const{inputs:e,backend:t,attrs:s}=n,{logits:r}=e,{dim:o}=s,i=de([o],r.shape),a=gc({inputs:{x:r},backend:t,attrs:{reductionIndices:i,keepDims:!1}}),c=je(a.shape,i),l=S({inputs:{x:a},backend:t,attrs:{shape:c}}),u=Cc({inputs:{a:r,b:l},backend:t}),d=fc({inputs:{x:u},backend:t}),h=as({inputs:{x:d},backend:t,attrs:{axis:i,keepDims:!1}}),f=S({inputs:{x:h},backend:t,attrs:{shape:c}}),p=xc({inputs:{a:d,b:f},backend:t});return t.disposeIntermediateTensorInfo(a),t.disposeIntermediateTensorInfo(l),t.disposeIntermediateTensorInfo(u),t.disposeIntermediateTensorInfo(d),t.disposeIntermediateTensorInfo(h),t.disposeIntermediateTensorInfo(f),p}const u$={kernelName:nd,backendName:"webgl",kernelFunc:bc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function d$(n){const{inputs:e,backend:t,attrs:s}=n,{logits:r}=e,{numSamples:o,seed:i,normalized:a}=s,c=a?r:bc({inputs:{logits:r},backend:t,attrs:{dim:r.shape.length-1}}),l=c.shape[0],u=c.shape[1],d=new o$(l,u,o),h=[[i]],f=t.runWebGLProgram(d,[c],"int32",h);return a||t.disposeIntermediateTensorInfo(c),f}const h$={kernelName:wu,backendName:"webgl",kernelFunc:d$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const f$=Ne+`
  return -x;
`,p$=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function m$(n){const{inputs:e,backend:t}=n,{x:s}=e;if(t.shouldExecuteOnCPU([s])){const o=t.texData.get(s.dataId),[i,a]=_x(o.values,s.shape,s.dtype);return t.makeTensorInfo(a,s.dtype,i)}let r;return y().getBool("WEBGL_PACK_UNARY_OPERATIONS")?r=new et(s.shape,p$):r=new Ue(s.shape,f$),t.runWebGLProgram(r,[s],s.dtype)}const g$={kernelName:yu,backendName:"webgl",kernelFunc:m$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const x$=Tf;function C$(n){Pe("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:r,scores:o}=e,{maxOutputSize:i,iouThreshold:a,scoreThreshold:c}=s,l=t.readSync(r.dataId),u=t.readSync(o.dataId),{selectedIndices:d}=x$(l,u,i,a,c);return t.makeTensorInfo([d.length],"int32",new Int32Array(d))}const b$={kernelName:vu,backendName:"webgl",kernelFunc:C$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const w$=Ef;function y$(n){Pe("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:r,scores:o}=e,{maxOutputSize:i,iouThreshold:a,scoreThreshold:c,padToMaxOutputSize:l}=s,u=t.readSync(r.dataId),d=t.readSync(o.dataId),{selectedIndices:h,validOutputs:f}=w$(u,d,i,a,c,l);return[t.makeTensorInfo([h.length],"int32",new Int32Array(h)),t.makeTensorInfo([],"int32",new Int32Array([f]))]}const $$={kernelName:Su,backendName:"webgl",kernelFunc:y$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const v$=Nf;function S$(n){Pe("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:e,backend:t,attrs:s}=n,{boxes:r,scores:o}=e,{maxOutputSize:i,iouThreshold:a,scoreThreshold:c,softNmsSigma:l}=s,u=t.readSync(r.dataId),d=t.readSync(o.dataId),h=i,f=a,p=c,x=l,{selectedIndices:g,selectedScores:m}=v$(u,d,h,f,p,x);return[t.makeTensorInfo([g.length],"int32",new Int32Array(g)),t.makeTensorInfo([m.length],"float32",new Float32Array(m))]}const I$={kernelName:Iu,backendName:"webgl",kernelFunc:S$};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class R${constructor(e,t,s,r){this.variableNames=["indices"],this.outputShape=[e,t],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${r}), float(${s}),
                      float(index == coords.y)));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const T$=n=>{const{inputs:e,backend:t,attrs:s}=n,{indices:r}=e,{dtype:o,depth:i,onValue:a,offValue:c}=s,l=E(r.shape),u=new R$(l,i,a,c),d=S({inputs:{x:r},backend:t,attrs:{shape:[l]}}),h=t.runWebGLProgram(u,[d],o);t.disposeIntermediateTensorInfo(d);const f=[...r.shape,i],p=S({inputs:{x:h},backend:t,attrs:{shape:f}});return t.disposeIntermediateTensorInfo(h),p},E$={kernelName:Tu,backendName:"webgl",kernelFunc:T$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qn(n){const{inputs:e,backend:t}=n,{x:s}=e;if(s.dtype==="complex64"){const r=Sn({inputs:{input:s},backend:t}),o=Qn({inputs:{x:r},backend:t}),i=cs({inputs:{input:s},backend:t}),a=Qn({inputs:{x:i},backend:t}),c=rt({inputs:{real:o,imag:a},backend:t});return t.disposeIntermediateTensorInfo(r),t.disposeIntermediateTensorInfo(o),t.disposeIntermediateTensorInfo(i),t.disposeIntermediateTensorInfo(a),c}else return In({attrs:{shape:s.shape,dtype:s.dtype,value:s.dtype==="string"?"":0},backend:t})}const N$={kernelName:Jo,backendName:"webgl",kernelFunc:Qn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wc(n){const{inputs:e,backend:t}=n,{x:s}=e;if(s.dtype==="string")throw new Error("onesLike is not supported under string dtype");if(s.dtype==="complex64"){const r=Sn({inputs:{input:s},backend:t}),o=wc({inputs:{x:r},backend:t}),i=cs({inputs:{input:s},backend:t}),a=Qn({inputs:{x:i},backend:t}),c=rt({inputs:{real:o,imag:a},backend:t});return t.disposeIntermediateTensorInfo(r),t.disposeIntermediateTensorInfo(o),t.disposeIntermediateTensorInfo(i),t.disposeIntermediateTensorInfo(a),c}else return In({attrs:{shape:s.shape,dtype:s.dtype,value:1},backend:t})}const k$={kernelName:Ru,backendName:"webgl",kernelFunc:wc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function A$(n){const{inputs:e,backend:t,attrs:s}=n,{axis:r}=s;if(e.length===1)return zs({inputs:{input:e[0]},backend:t,attrs:{dim:r}});const o=e[0].shape,i=e[0].dtype;e.forEach(u=>{No(o,u.shape,"All tensors passed to stack must have matching shapes"),I(i===u.dtype,()=>"All tensors passed to stack must have matching dtypes")});const a=[],c=e.map(u=>{const d=zs({inputs:{input:u},backend:t,attrs:{dim:r}});return a.push(d),d}),l=oc({inputs:c,backend:t,attrs:{axis:r}});return a.forEach(u=>t.disposeIntermediateTensorInfo(u)),l}const F$={kernelName:Eu,backendName:"webgl",kernelFunc:A$};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class D${constructor(e,t,s){this.variableNames=["x"],this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((l,u)=>l[0]+e[u]+l[1]);const r=e.length,o=U(r),i=t.map(l=>l[0]).join(","),a=t.map((l,u)=>l[0]+e[u]).join(","),c=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,r);if(r===1){this.userCode=`
        int start = ${i};
        int end = ${a};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${o} start = ${o}(${i});
      ${o} end = ${o}(${a});

      void main() {
        ${o} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${o} coords = outC - start;
          setOutput(getX(${c}));
        }
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class O${constructor(e,t,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"value",type:"float"}],this.outputShape=t.map((x,g)=>x[0]+e[g]+x[1]);const r=e.length,o=U(r),i=t.map(x=>x[0]).join(","),a=t.map((x,g)=>x[0]+e[g]).join(","),c=ie("rc",r),l=ie("source",r),u=`${c[r-1]} < ${this.outputShape[r-1]}`,d=r===1?"source":`vec2(${l.slice(-2).join()})`,h=[`${o} rc = outputLoc;`,`${c[r-1]} += 1;
       if(${u}) {
      `,r===1?"":`}
       rc = outputLoc;
       ${c[r-2]} += 1;
       if(${c[r-2]} < ${this.outputShape[r-2]}) {`,r===1?"":`  ${c[r-1]} += 1;
         if(${u}) {`],f=r===1?"rc < start || rc >= end":"any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))";let p="";for(let x=0,g=r===1?2:4;x<g;x++)p+=`
        ${h[x]}
        if (${f}) {
          result[${x}] = float(value);
        } else {
          ${o} source = rc - start;
          result[${x}] = getChannel(getX(${l.join()}), ${d});
        }
      `;p+=r===1?"} ":"}}",this.userCode=`
      const ${o} start = ${o}(${i});
      const ${o} end = ${o}(${a});

      void main() {
        ${o} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yc=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{paddings:o,constantValue:i}=s;if(E(r.shape)===0){const l=o.map((u,d)=>u[0]+r.shape[d]+u[1]);return In({backend:t,attrs:{shape:l,value:i,dtype:r.dtype}})}const a=y().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new O$(r.shape,o,i):new D$(r.shape,o,i),c=[[i]];return t.runWebGLProgram(a,[r],r.dtype,c)},P$={kernelName:Nu,backendName:"webgl",kernelFunc:yc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _$=`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,L$=`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+Et+`
  return result;
`,B$=te({opSnippet:_$,packedOpSnippet:L$}),M$={kernelName:Go,backendName:"webgl",kernelFunc:B$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function V$(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{axis:o,keepDims:i}=s,a=r.shape.length,c=[],l=de(o,r.shape);let u=l;const d=Te(u,a);let h=r;d!=null&&(h=ce({inputs:{x:r},backend:t,attrs:{perm:d}}),u=Ee(u.length,a),c.push(h)),Me("prod",u,a);let f;if(t.shouldExecuteOnCPU([h])){const p=t.texData.get(h.dataId).values,{outVals:x,outShape:g,outDtype:m}=Bx(h.shape,h.dtype,p,u);f=t.makeTensorInfo(g,m,x)}else{const[p,x]=He(h.shape,u),g=E(x),m=S({inputs:{x:h},backend:t,attrs:{shape:[-1,g]}}),C=Js(r.dtype),b=Nt(m,C,"prod",t);f=S({inputs:{x:b},backend:t,attrs:{shape:p}}),c.push(m),c.push(b)}if(i){c.push(f);const p=je(f.shape,l);f=S({inputs:{x:f},backend:t,attrs:{shape:p}})}return c.forEach(p=>t.disposeIntermediateTensorInfo(p)),f}const U$={kernelName:ku,backendName:"webgl",kernelFunc:V$};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function W$(n){const{inputs:e,backend:t,attrs:s}=n,{paramsNestedSplits:r,paramsDenseValues:o,indices:i}=e,{outputRaggedRank:a}=s,c=r.map(m=>t.readSync(m.dataId)),l=r.map(m=>m.shape),u=t.readSync(o.dataId),d=t.readSync(i.dataId),[h,f,p]=Mx(c,l,u,o.shape,o.dtype,d,i.shape,a),x=h.map(m=>t.makeTensorInfo([m.length],"int32",m)),g=t.makeTensorInfo(p,o.dtype,f);return x.concat([g])}const G$={kernelName:Au,backendName:"webgl",kernelFunc:W$};/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z$(n){const{inputs:e,backend:t}=n,{starts:s,limits:r,deltas:o}=e,i=t.readSync(s.dataId),a=t.readSync(r.dataId),c=t.readSync(o.dataId),[l,u]=Vx(i,s.shape,s.dtype,a,r.shape,c,o.shape),d=t.makeTensorInfo([l.length],"int32",l),h=t.makeTensorInfo([u.length],s.dtype,u);return[d,h]}const H$={kernelName:Fu,backendName:"webgl",kernelFunc:z$};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function X$(n){const{inputs:e,backend:t,attrs:s}=n,{shape:r,values:o,defaultValue:i,rowPartitionTensors:a}=e,{rowPartitionTypes:c}=s,l=t.readSync(r.dataId),u=t.readSync(o.dataId),d=t.readSync(i.dataId),h=a.map(g=>t.readSync(g.dataId)),f=a.map(g=>g.shape),[p,x]=Ux(l,r.shape,u,o.shape,o.dtype,d,i.shape,h,f,c);return t.makeTensorInfo(p,o.dtype,x)}const j$={kernelName:Du,backendName:"webgl",kernelFunc:X$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $c=n=>{const{backend:e,attrs:t}=n,{start:s,stop:r,step:o,dtype:i}=t,a=Wx(s,r,o,i);return e.makeTensorInfo([a.length],i,a)},q$={kernelName:Ou,backendName:"webgl",kernelFunc:$c};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const K$="return 1.0 / x;",Y$=_({opSnippet:K$}),Q$={kernelName:_u,backendName:"webgl",kernelFunc:Y$};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Z$=Ne+`
  return (x < 0.0) ? 0.0 : x;
`,J$=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,ev=_({opSnippet:Z$,packedOpSnippet:J$}),tv={kernelName:Ho,backendName:"webgl",kernelFunc:ev};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nv=Ne+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,sv=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,rv=_({opSnippet:nv,packedOpSnippet:sv}),ov={kernelName:jo,backendName:"webgl",kernelFunc:rv};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class iv{constructor(e,t,s,r,o){this.variableNames=["A"],this.outputShape=[];const[i,a,c,l]=e;this.outputShape=[i,t,s,l];const u=[r&&t>1?a-1:a,r&&s>1?c-1:c],d=[r&&t>1?t-1:t,r&&s>1?s-1:s];let h;o?h="(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)":h="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/d[0]},
          ${u[1]/d[1]});
      const vec2 inputShapeRC = vec2(${a}.0, ${c}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${h};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class av{constructor(e,t,s,r,o){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[i,a,c,l]=e;this.outputShape=[i,t,s,l];const u=[r&&t>1?a-1:a,r&&s>1?c-1:c],d=[r&&t>1?t-1:t,r&&s>1?s-1:s];let h;o?h="(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)":h="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/d[0]},
          ${u[1]/d[1]},
          ${u[1]/d[1]});
      const vec3 inputShapeRC = vec3(${a}.0, ${c}.0,
                                     ${c}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${h};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${s-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cv(n){const{inputs:e,backend:t,attrs:s}=n,{images:r}=e,{alignCorners:o,halfPixelCenters:i,size:a}=s,[c,l]=a,u=y().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new av(r.shape,c,l,o,i):new iv(r.shape,c,l,o,i);return t.runWebGLProgram(u,[r],"float32")}const lv={kernelName:Mu,backendName:"webgl",kernelFunc:cv};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class uv{constructor(e,t,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,r,o]=t,[,i,a]=e,c=[s&&i>1?r-1:r,s&&a>1?o-1:o],l=[s&&i>1?i-1:i,s&&a>1?a-1:a],u=c[0]/l[0],d=c[1]/l[1],h=1/u,f=1/d,p=Math.ceil(h)*2+2,x=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${d});

        const float invHeightScale = float(${h});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${x});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${i}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${a}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${r-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${o-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dv(n){const{inputs:e,backend:t,attrs:s}=n,{images:r,dy:o}=e,{alignCorners:i}=s,a=new uv(o.shape,r.shape,i);return t.runWebGLProgram(a,[o],o.dtype)}const hv={kernelName:Vu,backendName:"webgl",kernelFunc:dv};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fv{constructor(e,t,s,r,o){this.variableNames=["A"],this.outputShape=[];const[i,a,c,l]=e;this.outputShape=[i,t,s,l];const u=[r&&t>1?a-1:a,r&&s>1?c-1:c],d=[r&&t>1?t-1:t,r&&s>1?s-1:s],h=r?"0.5":"0.0";let f;o?f="max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))":f="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/d[0]},
          ${u[1]/d[1]});
      const vec2 inputShapeRC = vec2(${a}.0, ${c}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${h})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class pv{constructor(e,t,s,r,o){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[i,a,c,l]=e;this.outputShape=[i,t,s,l];const u=[r&&t>1?a-1:a,r&&s>1?c-1:c],d=[r&&t>1?t-1:t,r&&s>1?s-1:s],h=r?"0.5":"0.0";let f;o?f="max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))":f="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/d[0]},
          ${u[1]/d[1]},
          ${u[1]/d[1]});
      const vec3 inputShapeRC = vec3(${a}.0, ${c}.0,
                                     ${c}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${f};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${h})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${l-1};
        bool hasNextRow = coords.z < ${s-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mv(n){const{inputs:e,backend:t,attrs:s}=n,{images:r}=e,{alignCorners:o,halfPixelCenters:i,size:a}=s,[c,l]=a,u=y().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new pv(r.shape,c,l,o,i):new fv(r.shape,c,l,o,i);return t.runWebGLProgram(u,[r],r.dtype)}const gv={kernelName:Lu,backendName:"webgl",kernelFunc:mv};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class xv{constructor(e,t,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=t;const[,r,o]=t,[,i,a]=e,c=[s&&i>1?r-1:r,s&&a>1?o-1:o],l=[s&&i>1?i-1:i,s&&a>1?a-1:a],u=c[0]/l[0],d=c[1]/l[1],h=1/u,f=1/d,p=Math.ceil(h)*2+2,x=Math.ceil(f)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${d});

        const float invHeightScale = float(${h});
        const float invWidthScale = float(${f});

        const int winHeight = int(${p});
        const int winWidth = int(${x});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${i}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${a}) {
              continue;
            }

            float sourceFracRow =
              float(${c[0]}) *
                (float(dyR) / float(${l[0]}));

            float sourceFracCol =
                float(${c[1]}) *
                  (float(dyC) / float(${l[1]}));

            int sourceNearestRow = int(min(
                float(int(${r}) - 1),
                ${s} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${o}) - 1),
                ${s} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cv(n){const{inputs:e,backend:t,attrs:s}=n,{images:r,dy:o}=e,{alignCorners:i}=s,a=new xv(o.shape,r.shape,i);return t.runWebGLProgram(a,[o],o.dtype)}const bv={kernelName:Bu,backendName:"webgl",kernelFunc:Cv};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wv{constructor(e,t){this.variableNames=["x"];const s=e.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);if(this.outputShape=e,s===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${e[0]} - coord - 1));
        }
      `;return}const r=a=>t.indexOf(a)!==-1&&e[a]!==1?`${e[a]} - coords[${a}] - 1`:`coords[${a}]`,o=e.map((a,c)=>r(c)).join(","),i=U(s);this.userCode=`
      void main() {
        ${i} coords = getOutputCoords();
        setOutput(getX(${o}));
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class yv{constructor(e,t){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0;const s=e.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);this.outputShape=e;const r=ie("rc",s),o=`${r[s-1]} + 1 < ${this.outputShape[s-1]}`,i=`${r[s-2]} + 1 < ${this.outputShape[s-2]}`,a=U(s);s===1?this.userCode=`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${e[0]} - rc - 1),
            ${e[0]} - rc - 1);
          if(${o}){
              result.g = getChannel(getX(${e[0]} - (rc  + 1) - 1),
                ${e[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:this.userCode=`
        void main() {
          ${a} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${c(r.slice())};
          if(${o}){
            result.g = ${l(r.slice())};
          }
          if(${i}) {
            result.b = ${u(r.slice())};
            if(${o}) {
              result.a = ${d(r.slice())};
            }
          }
          setOutput(result);
        }
    `;function c(p){return h(p)}function l(p){return p[s-1]="("+p[s-1]+" + 1)",h(p)}function u(p){return p[s-2]="("+p[s-2]+" + 1)",h(p)}function d(p){return p[s-1]="("+p[s-1]+" + 1)",p[s-2]="("+p[s-2]+" + 1)",h(p)}function h(p){const x=e.map((C,b)=>f(b,p)),g=x.join(","),m=x.slice(-2).join(",");return`getChannel(getX(${g}), vec2(${m}))`}function f(p,x){return t.indexOf(p)!==-1&&e[p]!==1?`${e[p]} - ${x[p]} - 1`:`${x[p]}`}}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $v(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{dims:o}=s,i=r.shape.length,a=de(o,r.shape);if(i===0)return ge({inputs:{x:r},backend:t});const c=y().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new yv(r.shape,a):new wv(r.shape,a);return t.runWebGLProgram(c,[r],r.dtype)}const vv={kernelName:Uu,backendName:"webgl",kernelFunc:$v};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Sv{constructor(e,t){this.variableNames=["Image"],this.outputShape=[],this.customUniforms=[{name:"params",type:"vec4"}];const s=e[1],r=e[2];this.outputShape=e;let o="";typeof t=="number"?o=`float outputValue = ${t.toFixed(2)};`:o=`
        vec3 fill = vec3(${t.join(",")});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${o}
          if(coordX >= 0 && coordX < ${r} && coordY >= 0 && coordY < ${s}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Iv={kernelName:Sd,backendName:"webgl",kernelFunc:({inputs:n,attrs:e,backend:t})=>{const{image:s}=n,{radians:r,fillValue:o,center:i}=e,a=t,c=new Sv(s.shape,o),[l,u]=Ji(i,s.shape[1],s.shape[2]),d=[[l,u,Math.sin(r),Math.cos(r)]];return a.runWebGLProgram(c,[s],s.dtype,d)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rv=`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`,Tv=_({opSnippet:Rv}),Ev={kernelName:Wu,backendName:"webgl",kernelFunc:Tv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nv="return inversesqrt(x);",kv=_({opSnippet:Nv,cpuKernelImpl:Gx}),Av={kernelName:Gu,backendName:"webgl",kernelFunc:kv};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class yr{constructor(e,t,s,r,o,i,a=!0,c=!1){this.variableNames=["updates","indices","defaultValue"],this.outputShape=i;const l=U(o.length),u=U(i.length);let d="";s===1?d="i":s===2&&(d="i, j");const h=`getIndices(${d})`;let f="";r===1?f="i":r===2&&(f="i, coords[1]");const p=`getUpdates(${f})`;let x="";c&&(x="coords[0], coords[1]");const g=`getDefaultValue(${x})`,m=t>1?"strides[j]":"strides";this.userCode=`
        ${l} strides = ${l}(${o});

        void main() {
          ${u} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${e}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${t}; j++) {
              int index = round(${h});
              flattenedIndex += index * ${m};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${p};
              found = true;
            }
          }
          setOutput(mix(${g}, sum, float(found)));
        }
      `}}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Fv{constructor(e,t,s,r,o,i,a=!0,c=!1){this.variableNames=["updates","indices","defaultValue"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=i;const l=U(o.length),u=U(i.length);let d="";s===1?d="i":s===2&&(d="i, j");const h=`getIndices(${d})`;let f="";r===1?f="i":r===2&&(f="i, coords[1]");const p=`getUpdates(${f})`;let x="";c&&(x="coords[0], coords[1]");const g=`getDefaultValue(${x})`,m=t>1?"strides[j]":"strides",C=t>1?"strides[j + 1]":"strides";this.userCode=`
        ${l} strides = ${l}(${o});

        void main() {
          ${u} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${e}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${t}; j+=2) {
              ivec4 index = round(${h});
              flattenedIndex += index.xz * ${m};
              if (j + 1 < ${t}) {
                flattenedIndex += index.yw * ${C};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${p};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${g}, sum, found));
        }
      `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dv(n){const{inputs:e,backend:t,attrs:s}=n,{indices:r,updates:o}=e,{shape:i}=s,{sliceRank:a,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=ns(o,r,i),h=[d/l,l];if(d===0)return t.makeTensorInfo(i,r.dtype);const f=S({inputs:{x:r},backend:t,attrs:{shape:[c,a]}}),p=S({inputs:{x:o},backend:t,attrs:{shape:[c,l]}}),x=t.makeTensorInfo([],"float32",new Float32Array([0]));let g;y().getBool("WEBGL_PACK")?g=new Fv(c,a,f.shape.length,p.shape.length,u,h):g=new yr(c,a,f.shape.length,p.shape.length,u,h);const m=t.runWebGLProgram(g,[p,f,x],p.dtype),C=S({inputs:{x:m},backend:t,attrs:{shape:i}});return t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(m),t.disposeIntermediateTensorInfo(x),C}const Ov={kernelName:zu,backendName:"webgl",kernelFunc:Dv};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pv{constructor(e,t,s,r){this.variableNames=["sortedSequence","values"],this.customUniforms=[{name:"numInputs",type:"int"}],this.outputShape=[e,s];const o="while (left < right) {",i=`for (int i = 0; i < ${Math.ceil(Math.log2(t+1))}; ++i) { if (left >= right) break;`,a=y().getNumber("WEBGL_VERSION")===2?o:i,c=r==="left"?"<":"<=";this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${a}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${c} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _v(n){const{inputs:e,backend:t,attrs:s}=n,{sortedSequence:r,values:o}=e,{side:i}=s,a=new Pv(r.shape[0],r.shape[1],o.shape[1],i),c=[[r.shape[1]]];return t.runWebGLProgram(a,[r,o],"int32",c)}const Lv={kernelName:Xu,backendName:"webgl",kernelFunc:_v};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bv{constructor(e,t,s){this.variableNames=["c","a","b"],this.outputShape=t;let r,o;if(s>4)throw Error(`Where for rank ${s} is not yet supported`);if(s===1)o="resRC",r="resRC";else{const a=["resRC.x","resRC.y","resRC.z","resRC.w"],c=[],l=[];for(let u=0;u<t.length;u++)l.push(`${a[u]}`),u<e&&c.push(`${a[u]}`);r=c.join(),o=l.join()}const i=U(s);this.userCode=`
      void main() {
        ${i} resRC = getOutputCoords();
        float cVal = getC(${r});
        if (cVal >= 1.0) {
          setOutput(getA(${o}));
        } else {
          setOutput(getB(${o}));
        }
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mv(n){const{inputs:e,backend:t}=n,{condition:s,t:r,e:o}=e,i=new Bv(s.shape.length,r.shape,r.shape.length);return t.runWebGLProgram(i,[s,r,o],ze(r.dtype,o.dtype))}const Vv={kernelName:ju,backendName:"webgl",kernelFunc:Mv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uv=`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${na};
  float scale = ${sa};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`,Wv=_({opSnippet:Uv}),Gv={kernelName:qu,backendName:"webgl",kernelFunc:Wv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zv=nn+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,Hv=`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Xv=_({opSnippet:zv,packedOpSnippet:Hv,cpuKernelImpl:Hx}),jv={kernelName:qo,backendName:"webgl",kernelFunc:Xv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qv=`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`,Kv=_({opSnippet:qv}),Yv={kernelName:Zu,backendName:"webgl",kernelFunc:Kv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qv=nn+`
  return sin(x);
`,Zv=`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${Et}
  return result;
`,Jv=_({opSnippet:Qv,packedOpSnippet:Zv}),eS={kernelName:Yu,backendName:"webgl",kernelFunc:Jv};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tS=`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`,nS=_({opSnippet:tS}),sS={kernelName:Qu,backendName:"webgl",kernelFunc:nS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rS=`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`,oS=_({opSnippet:rS}),iS={kernelName:Ju,backendName:"webgl",kernelFunc:oS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const aS=n=>{const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{blockShape:o,paddings:i}=s;I(r.shape.length<=4,()=>"spaceToBatchND for rank > 4 with a WebGL backend not implemented yet");const a=o.reduce((m,C)=>m*C),c=[[0,0]];c.push(...i);for(let m=1+o.length;m<r.shape.length;++m)c.push([0,0]);const l=[],u=yc({inputs:{x:r},backend:t,attrs:{paddings:c,constantValue:0}}),d=ur(u.shape,o,a,!1),h=dr(d.length,o.length,!1),f=hr(u.shape,o,a,!1),p=S({inputs:{x:u},backend:t,attrs:{shape:d}}),x=ce({inputs:{x:p},backend:t,attrs:{perm:h}}),g=S({inputs:{x},backend:t,attrs:{shape:f}});return l.push(u),l.push(p),l.push(x),l.forEach(m=>t.disposeIntermediateTensorInfo(m)),g},cS={kernelName:ed,backendName:"webgl",kernelFunc:aS};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lS(n){const{inputs:e,backend:t}=n,{indices:s,values:r,denseShape:o,defaultValue:i}=e;if(o.shape.length!==1)throw new Error(`Dense shape must be a vector, saw:
         ${o.shape}`);if(s.shape.length!==2)throw new Error(`Indices must be a matrix, saw:
         ${s.shape}`);if(r.shape.length!==1)throw new Error(`Values must be a vector, saw:
         ${r.shape}`);if(i.shape.length!==0)throw new Error(`Default value must be a scalar, saw:
        ${i.shape}`);const a=t.readSync(s.dataId),c=t.readSync(r.dataId),l=t.readSync(o.dataId),u=t.readSync(i.dataId)[0],[d,h,f,p,x]=jx(a,s.shape,s.dtype,c,r.dtype,l,u);return[t.makeTensorInfo(h,s.dtype,d),t.makeTensorInfo([h[0]],r.dtype,f),t.makeTensorInfo([p.length],"bool",new Uint8Array(p.map(g=>Number(g)))),t.makeTensorInfo([x.length],s.dtype,new Int32Array(x))]}const uS={kernelName:sd,backendName:"webgl",kernelFunc:lS};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dS(n){const{inputs:e,backend:t}=n,{inputIndices:s,inputShape:r,newShape:o}=e;if(s.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape ${s.shape}`);if(r.shape.length!==1)throw new Error(`Input shape should be a vector but received shape ${r.shape}`);if(o.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${o.shape}`);const i=Array.from(t.readSync(r.dataId)),a=t.readSync(s.dataId),c=Array.from(t.readSync(o.dataId)),[l,u,d]=qx(a,s.shape,s.dtype,i,c);return[t.makeTensorInfo(u,s.dtype,l),t.makeTensorInfo([d.length],o.dtype,new Int32Array(d))]}const hS={kernelName:rd,backendName:"webgl",kernelFunc:dS};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fS(n){const{inputs:e,backend:t}=n,{data:s,indices:r,segmentIds:o}=e;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.shape.length!==1)throw new Error(`Indices should be a vector but received shape
              ${r.shape}`);if(o.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
              ${o.shape}`);const i=t.readSync(s.dataId),a=t.readSync(r.dataId),c=t.readSync(o.dataId),[l,u]=Xa(i,s.shape,s.dtype,a,c,!0);return t.makeTensorInfo(u,s.dtype,l)}const pS={kernelName:od,backendName:"webgl",kernelFunc:fS};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mS(n){const{inputs:e,backend:t}=n,{data:s,indices:r,segmentIds:o}=e;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.shape.length!==1)throw new Error(`Indices should be a vector but received shape
             ${r.shape}`);if(o.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
             ${o.shape}`);const i=t.readSync(s.dataId),a=t.readSync(r.dataId),c=t.readSync(o.dataId),[l,u]=Xa(i,s.shape,s.dtype,a,c);return t.makeTensorInfo(u,s.dtype,l)}const gS={kernelName:id,backendName:"webgl",kernelFunc:mS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xS(n){const{inputs:e,backend:t,attrs:s}=n,{sparseIndices:r,sparseValues:o,defaultValue:i}=e,{outputShape:a}=s,{sliceRank:c,numUpdates:l,sliceSize:u,strides:d,outputSize:h}=ns(o,r,a),f=!1;if(o.dtype==="string"){const m=t.bufferSync(r),C=t.bufferSync(o),b=Vt(t.readSync(i.dataId)[0]),w=zx(m,C,a,h,u,l,c,d,b,f);return t.makeTensorInfo(a,w.dtype,w.values)}const p=new yr(l,c,r.shape.length,o.shape.length,d,[h,1],f),x=t.runWebGLProgram(p,[o,r,i],o.dtype),g=S({inputs:{x},backend:t,attrs:{shape:a}});return t.disposeIntermediateTensorInfo(x),g}const CS={kernelName:ad,backendName:"webgl",kernelFunc:xS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bS(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{numOrSizeSplits:o,axis:i}=s,a=de(i,r.shape)[0],c=ma(r,o,a),l=r.shape.length,u=new Array(l).fill(0),d=r.shape.slice();return c.map(h=>{const f=[...d];f[a]=h;const p=sn({inputs:{x:r},backend:t,attrs:{begin:u,size:f}});return u[a]+=h,p})}const wS={kernelName:td,backendName:"webgl",kernelFunc:bS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Io="return sqrt(x);",yS=_({opSnippet:Io,packedOpSnippet:Io,cpuKernelImpl:Kx}),$S={kernelName:Ko,backendName:"webgl",kernelFunc:yS};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vS="return x * x;",SS=_({opSnippet:vS}),IS={kernelName:ld,backendName:"webgl",kernelFunc:SS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ro="return (a - b) * (a - b);",RS=te({opSnippet:Ro,packedOpSnippet:Ro}),TS={kernelName:cd,backendName:"webgl",kernelFunc:RS};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ES(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e;if(r.dtype!=="string")throw new Error("Input must be of datatype string");const o=t.readSync(r.dataId),i=Gt(o),a=Yx(i,"string",s);return t.makeTensorInfo(r.shape,"string",a)}const NS={kernelName:ud,backendName:"webgl",kernelFunc:ES};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kS({inputs:n,attrs:e,backend:t}){const{x:s}=n,r=Ne+`
    return x > 0.0 ? 1.0 : float(${e.alpha});
  `,o=new Ue(s.shape,r);return t.runWebGLProgram(o,[s],s.dtype)}const AS={kernelName:ei,backendName:"webgl",kernelFunc:kS};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class FS{constructor(e,t,s){this.variableNames=["x"],this.outputShape=s;const r=s.length,o=U(s.length),i=U(s.length);let a="";if(r===1)a="coords * strides + begin";else{let c=0;a=s.map((l,u)=>(c++,s.length===1?`coords * strides[${u}] + begin[${u}]`:`coords[${c-1}] * strides[${u}] + begin[${u}]`)).join(",")}this.userCode=`
      ${o} begin = ${o}(${e});
      ${o} strides = ${o}(${t});

      void main() {
        ${i} coords = getOutputCoords();
        setOutput(getX(${a}));
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function DS(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{begin:o,end:i,strides:a,beginMask:c,endMask:l,ellipsisMask:u,newAxisMask:d,shrinkAxisMask:h}=s,{finalShapeSparse:f,finalShape:p,isIdentity:x,sliceDim0:g,isSimpleSlice:m,begin:C,end:b,strides:w}=ji(r.shape,o,i,a,c,l,u,d,h);let $;if(x)$=S({inputs:{x:r},backend:t,attrs:{shape:p}});else if(g||m){I(r.shape.length>=1,()=>`Input must have rank at least 1, got: ${r.shape.length}`);const T=Li(C,b,w),v=sn({inputs:{x:r},backend:t,attrs:{begin:C,size:T}});$=S({inputs:{x:v},backend:t,attrs:{shape:p}}),t.disposeIntermediateTensorInfo(v)}else if(t.shouldExecuteOnCPU([r])){const v=t.readSync(r.dataId),D=ee(r.shape,r.dtype,v),O=Qx(f,D,w,C);$=t.makeTensorInfo(p,r.dtype,O.values)}else{const v=new FS(C,w,f);$=t.runWebGLProgram(v,[r],r.dtype)}const N=S({inputs:{x:$},backend:t,attrs:{shape:p}});return t.disposeIntermediateTensorInfo($),N}const OS={kernelName:dd,backendName:"webgl",kernelFunc:DS};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function PS(n){const{inputs:e,backend:t,attrs:s}=n,{separator:r,nGramWidths:o,leftPad:i,rightPad:a,padWidth:c,preserveShortSequences:l}=s,{data:u,dataSplits:d}=e,h=t.readSync(u.dataId),f=t.readSync(d.dataId),[p,x]=Zx(h,f,r,o,i,a,c,l);return[t.makeTensorInfo([p.length],"string",p),t.makeTensorInfo(d.shape,"int32",x)]}const _S={kernelName:hd,backendName:"webgl",kernelFunc:PS};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function LS(n){const{inputs:e,backend:t,attrs:s}=n,{skipEmpty:r}=s,{input:o,delimiter:i}=e;if(o.dtype!=="string")throw new Error("Input must be of datatype string");if(o.shape.length!==1)throw new Error(`Input must be a vector, got shape: ${o.shape}`);if(i.shape.length!==0)throw new Error(`Delimiter must be a scalar, got shape: ${i.shape}`);const a=t.readSync(o.dataId),c=t.readSync(i.dataId)[0],[l,u,d]=Jx(a,c,r),h=u.length;return[t.makeTensorInfo([h,2],"int32",l),t.makeTensorInfo([h],"string",u),t.makeTensorInfo([2],"int32",new Int32Array(d))]}const BS={kernelName:fd,backendName:"webgl",kernelFunc:LS};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function MS(n){const{inputs:e,backend:t,attrs:s}=n,{numBuckets:r}=s,{input:o}=e;if(o.dtype!=="string")throw new Error("Input must be of datatype string");if(r<=0)throw new Error("Number of buckets must be at least 1");const i=t.readSync(o.dataId),a=e0(i,r);return t.makeTensorInfo(o.shape,"int32",a)}const VS={kernelName:pd,backendName:"webgl",kernelFunc:MS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const US="return tan(x);",WS=_({opSnippet:US}),GS={kernelName:md,backendName:"webgl",kernelFunc:WS};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zS=`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`,HS=_({opSnippet:zS}),XS={kernelName:gd,backendName:"webgl",kernelFunc:HS};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jS(n){const{inputs:e,backend:t,attrs:s}=n,{tensor:r,indices:o,updates:i}=e,{sliceRank:a,numUpdates:c,sliceSize:l,strides:u,outputSize:d}=ns(i,o,r.shape),h=[d/l,l];if(d===0)return t.makeTensorInfo(r.shape,o.dtype);const f=S({inputs:{x:o},backend:t,attrs:{shape:[c,a]}}),p=S({inputs:{x:i},backend:t,attrs:{shape:[c,l]}}),x=S({inputs:{x:r},backend:t,attrs:{shape:h}}),g=new yr(c,a,f.shape.length,p.shape.length,u,h,!1,!0),m=t.runWebGLProgram(g,[p,f,x],x.dtype),C=S({inputs:{x:m},backend:t,attrs:{shape:r.shape}});return t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(p),t.disposeIntermediateTensorInfo(x),t.disposeIntermediateTensorInfo(m),C}const qS={kernelName:Hu,backendName:"webgl",kernelFunc:jS};/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class KS{constructor(e,t){this.variableNames=["A"];const s=new Array(e.length);for(let i=0;i<s.length;i++)s[i]=e[i]*t[i];this.outputShape=s,this.rank=s.length;const r=U(this.rank),o=YS(e);this.userCode=`
      void main() {
        ${r} resRC = getOutputCoords();
        setOutput(getA(${o}));
      }
    `}}function YS(n){const e=n.length;if(e>5)throw Error(`Tile for rank ${e} is not yet supported`);if(e===1)return`imod(resRC, ${n[0]})`;const t=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u"],s=[];for(let r=0;r<n.length;r++)s.push(`imod(${t[r]}, ${n[r]})`);return s.join()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vc(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{reps:o}=s;if(r.dtype==="string"||r.shape.length>5){const c=t.readSync(r.dataId),l=r.dtype==="string"?c.map(h=>Vt(h)):c,u=ee(r.shape,r.dtype,l),d=n0(u,o);return t.makeTensorInfo(d.shape,d.dtype,d.values)}const i=new KS(r.shape,o);return t.runWebGLProgram(i,[r],r.dtype)}const QS={kernelName:Zo,backendName:"webgl",kernelFunc:vc};class ZS{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"negativeInf",type:"float"},{name:"dir",type:"int"},{name:"inc",type:"int"}],this.outputShape=e,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}}class JS{constructor(e){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"k",type:"int"}],this.outputShape=e,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ot(n,e){e!==null&&n.disposeIntermediateTensorInfo(e)}function To(n){let e=1;for(;e<n;)e*=2;return e}function eI(n){const{inputs:e,backend:t,attrs:s}=n,{x:r}=e,{k:o,sorted:i}=s,a=y().getNumber("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD"),c=y().getNumber("TOPK_K_CPU_HANDOFF_THRESHOLD"),l=r.shape,u=l[l.length-1];if(t.shouldExecuteOnCPU([r])||u<a||o>c){const O=t.readSync(r.dataId),[L,M]=s0(O,l,r.dtype,o,i);return[t.makeTensorInfo(L.shape,L.dtype,L.values),t.makeTensorInfo(M.shape,M.dtype,M.values)]}if(o===0)return l[l.length-1]=0,[t.makeTensorInfo(l,r.dtype,[]),t.makeTensorInfo(l,"int32",[])];if(u===1)return[r,In({attrs:{shape:l,dtype:"int32",value:0},backend:t})];const d=t.texData.get(r.dataId),h=d!==null&&d.isPacked,f=h?t.unpackTensor(r):r,x=E(l)/u,g=S({inputs:{x:f},attrs:{shape:[x,u]},backend:t});h&&ot(t,f);const m=To(o),C=To(u);let b=null;const w=()=>b===null?[g,g]:[g,b],$=(O,L,M)=>{const fe=w(),K=new ZS(M),we=[[u],[b===null?1:0],[Number.NEGATIVE_INFINITY],[O],[L]],ke=b;b=t.runWebGLProgram(K,fe,"int32",we),ot(t,ke)};for(let O=1;O<m;O*=2){const L=O*2;for(let M=O;M>=1;M/=2)$(L,M,[x,C])}for(let O=C;O>m;O/=2){const L=w(),M=new JS([x,O/2]),K=[[u],[b===null?1:0],[m]],ne=b;b=t.runWebGLProgram(M,L,"int32",K),ot(t,ne);const we=m/2,ke=we*2;for(let re=we;re>=1;re/=2)$(ke,re,b.shape)}let N=b;b=sn({inputs:{x:b},backend:t,attrs:{begin:0,size:[x,o]}}),ot(t,N);let T=mc({inputs:{x:g,indices:b},backend:t,attrs:{axis:1,batchDims:1}});ot(t,g);const v=l.slice(0,-1);v.push(o),N=b,b=S({inputs:{x:b},attrs:{shape:v},backend:t}),ot(t,N);const D=T;return T=S({inputs:{x:T},attrs:{shape:v},backend:t}),ot(t,D),[T,b]}const tI={kernelName:xd,backendName:"webgl",kernelFunc:eI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class nI{constructor(e,t,s,r,o,i){this.variableNames=["Image","Transforms"],this.outputShape=i;const a=s==="nearest"?1:2;let c;switch(r){case"constant":c=1;break;case"reflect":c=2;break;case"wrap":c=3;break;case"nearest":c=4;break;default:c=1;break}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${c} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${c} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${c} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${e} && 0 <= coordX && coordX < ${t}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${o});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${o});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${t}));
                float mapY = mapCoord(inY, float(${e}));

                if (${a} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sI(n){const{inputs:e,backend:t,attrs:s}=n,{image:r,transforms:o}=e,{interpolation:i,fillMode:a,fillValue:c,outputShape:l}=s,[u,d,h,f]=r.shape,[p,x]=l??[d,h],g=[u,p,x,f],m=new nI(d,h,i,a,c,g);return t.runWebGLProgram(m,[r,o],"float32")}const rI={kernelName:Cd,backendName:"webgl",kernelFunc:sI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oI(n){const{inputs:e,attrs:t,backend:s}=n,{axis:r}=t,{x:o}=e;$n(o,"unique"),console.warn("WARNING: ","UI might be locked temporarily as data is being downloaded");const i=s.readSync(o.dataId),{outputValues:a,outputShape:c,indices:l}=r0(i,r,o.shape,o.dtype);return[s.makeTensorInfo(c,o.dtype,a),s.makeTensorInfo([l.length],"int32",l)]}const iI={kernelName:wd,backendName:"webgl",kernelFunc:oI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aI(n){const{inputs:e,backend:t,attrs:s}=n,{value:r}=e;let{axis:o}=s;o<0&&(o+=r.shape.length);const i=r,a=i.shape.length,c=r.shape[o],l=new Array(a-1);let u=0;for(let x=0;x<a;x++)x!==o&&(l[u++]=i.shape[x]);const d=[],h=new Array(a).fill(0),f=i.shape.slice();f[o]=1;const p=new Array(c);for(let x=0;x<p.length;x++){h[o]=x;const g=sn({inputs:{x:i},backend:t,attrs:{begin:h,size:f}}),m=S({inputs:{x:g},backend:t,attrs:{shape:l}});p[x]=m,d.push(g)}return d.forEach(x=>t.disposeIntermediateTensorInfo(x)),p}const cI={kernelName:yd,backendName:"webgl",kernelFunc:aI};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class lI{constructor(e,t){this.variableNames=["x","segmentIds"];const s=e.windowSize,r=e.batchSize,o=e.inSize,i=e.numSegments,a=i*Math.ceil(o/s);this.outputShape=[r,a];const c="0.0",l="sumValue",u=Math.floor(s/4)*4,d=s%4,h=`
        sumValue += dot(values, segFilter);
    `;let f="";o%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${o}) {
          return initializationValue;
        }
      `);let p="";o%s>0&&(p=`
        if (inIdx < 0 || inIdx >= ${o}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = ${c};

      float getValue(int batch, int inIdx) {
        ${f}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${p}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${i})) * float(${s}));
        int currentSeg = int(mod(float(outIdx), float(${i})));

        float sumValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${h}
        }

        int inIdx = inOffset + ${u};
        if (${d===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${h}
        } else if (${d===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${h}
        } else if (${d===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${h}
        }
        setOutput(${l});
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uI(n){const{inputs:e,backend:t,attrs:s}=n,{x:r,segmentIds:o}=e,{numSegments:i}=s,a=r.shape.length,c=[];let l=0;const u=Te([l],a);let d=r;u!=null&&(d=ce({inputs:{x:r},backend:t,attrs:{perm:u}}),c.push(d),l=Ee(1,a)[0]);const h=Ea(d.shape,l,i),f=E([d.shape[l]]),p=S({inputs:{x:d},backend:t,attrs:{shape:[-1,f]}});c.push(p);const x=Js(r.dtype),g=(w,$,N,T,v)=>{const D=w.shape[0],O=w.shape[1],L=Ta(O,v),M={windowSize:L,inSize:O,batchSize:D,numSegments:v},fe=new lI(M,$),K=t.compileAndRun(fe,[w,N],T);if(c.push(K),K.shape[1]===v)return K;const ne=$c({backend:t,attrs:{start:0,stop:v,step:1,dtype:"float32"}}),we=vc({inputs:{x:ne},backend:t,attrs:{reps:[O/L]}});return c.push(ne),c.push(we),g(K,$,we,T,v)},m=g(p,"unsortedSegmentSum",o,x,i),C=S({inputs:{x:m},backend:t,attrs:{shape:h}});let b=C;if(u!=null){c.push(C);const w=rr(u);b=ce({inputs:{x:b},backend:t,attrs:{perm:w}})}return c.forEach(w=>t.disposeIntermediateTensorInfo(w)),b}const dI={kernelName:$d,backendName:"webgl",kernelFunc:uI};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hI=[K0,Q0,eC,sC,oC,cC,uC,hC,gC,CC,yC,SC,TC,AC,OC,_C,BC,WC,zC,XC,YC,sb,ob,lb,db,xb,bb,vb,F0,Rb,Ab,Pb,Ub,zb,Xb,qb,Yb,ew,sw,iw,cw,uw,hw,mw,xw,yw,vw,Rw,Nw,Aw,Pw,Mw,Gw,Xw,Kw,Yw,Zw,e1,n1,r1,i1,u1,f1,g1,C1,y1,S1,E1,F1,A0,O1,Nb,L1,V1,G1,O0,j1,Q1,J1,sy,iy,uy,fy,xy,yy,Sy,Ry,ky,Fy,Oy,By,Vy,Wy,zy,Xy,Yy,e$,r$,h$,L0,g$,b$,$$,I$,fb,E$,k$,F$,P$,M$,_0,U$,G$,H$,j$,q$,pb,c$,Q$,tv,ov,M0,lv,hv,gv,bv,vv,Iv,Ev,Av,Ov,Lv,Vv,Gv,jv,Yv,eS,sS,tb,u$,iS,cS,uS,hS,pS,gS,CS,wS,$S,IS,TS,NS,AS,OS,_S,BS,VS,l$,X0,GS,XS,qS,QS,tI,rI,j0,iI,cI,dI,N$];for(const n of hI)kd(n);export{mI as r,pI as s};
