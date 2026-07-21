(* ::Package:: *)

<<EDCRGTCcode.m


(* ::Input::Initialization:: *)
$Assumptions=l\[Element]Integers&&l>=0 && Pi>=theta>=0;


(* ::Input::Initialization:: *)
x={t,r,theta,phi};


(* ::Input::Initialization:: *)
gbg=DiagonalMatrix[{-(1-2M/r),1/(1-2M/r),r^2,r^2*Sin[theta]^2}];


(* ::Subsection:: *)
(*Tensor Spherical Harmonics*)


(* ::Input::Initialization:: *)
Clear[h,h1,h0]


(* ::Input::Initialization:: *)
\[Gamma]Ang=DiagonalMatrix[{1,Sin[theta]^2}];


(* ::Input::Initialization:: *)
xAng={theta,phi};


(* ::Input::Initialization:: *)
RGtensors[\[Gamma]Ang,xAng];


(* ::Input:: *)
(*(*PS: the last index of covD[xx] is the derivative index*)*)


(* ::Input:: *)
(*(*\nabla Y_{lm}, the parity even vector spherical harmonics*)*)


(* ::Input::Initialization:: *)
\[Epsilon]Ang=Sin[theta]*{{0,1},{-1,0}};


(* ::Input::Initialization:: *)
Sa=multiDot[\[Epsilon]Ang,multiDot[gUU,covD[LegendreP[l,Cos[theta]]],{2,1}],{2,1}]//Simplify


(* ::Input:: *)
(*(*Verify the Laplacian equation of Sa in 2d: \nabla^2 Sa=-(l(l+1)-1)Sa*)*)


(* ::Input::Initialization:: *)
hoddvec={{0,0,h0[r]*Sa[[1]],h0[r]*Sa[[2]]},{0,0,h1[r]*Sa[[1]],h1[r]*Sa[[2]]},{h0[r]*Sa[[1]],h1[r]*Sa[[1]],0,0},{h0[r]*Sa[[2]],h1[r]*Sa[[2]],0,0}};


(* ::Input::Initialization:: *)
Stenab=(covD[Sa]+Transpose[covD[Sa]])/2//FullSimplify;


(* ::Input::Initialization:: *)
Zero2=ConstantArray[0,{2,2}];


(* ::Input::Initialization:: *)
hoddten=ArrayFlatten[{{Zero2,Zero2},{Zero2,h[r]*Stenab}}]


(* ::Input::Initialization:: *)
hoddfull=hoddvec+hoddten;


(* ::Input::Initialization:: *)
Stenfull=ArrayFlatten[{{Zero2,Zero2},{Zero2,Stenab}}];


\[Epsilon]tr={{0,1},{-1,0}};


\[Epsilon]inv={{0,-1},{1,0}};


gtr=DiagonalMatrix[{-(1-2M/r),1/(1-2M/r)}];


xtr={t,r};


RGtensors[gtr,xtr];


(* ::Input::Initialization:: *)
\[Psi]t[r_,t_]:=r^3*multiDot[\[Epsilon]inv,covD[{k0[r,t],k1[r,t]}/r^2],{1,1},{2,2}]//FullSimplify


(* ::Input::Initialization:: *)
Y[y_]:=LegendreP[l,y]


(* ::Input::Initialization:: *)
\[Gamma]Angy=DiagonalMatrix[{1/(1-y^2),(1-y^2)}];


(* ::Input::Initialization:: *)
xAngy={y,phi};


(* ::Input::Initialization:: *)
RGtensors[\[Gamma]Angy,xAngy];


(* ::Input::Initialization:: *)
\[Epsilon]Angy={{0,-1},{1,0}};


(* ::Input::Initialization:: *)
Say=multiDot[\[Epsilon]Angy,multiDot[gUU,covD[Y[y]],{2,1}],{2,1}]//Simplify;


(* ::Input::Initialization:: *)
Stenaby=(covD[Say]+Transpose[covD[Say]])/2//FullSimplify;


(* ::Input::Initialization:: *)
recurrence=LegendreP[l+2,y]->((2 l+3) y LegendreP[l+1,y]-(l+1) LegendreP[l,y])/(l+2);


(* ::Input::Initialization:: *)
Stenaby=Stenaby/.recurrence//Simplify;


(* ::Subsection:: *)
(*Static Perturbative Equation*)


(* ::Input::Initialization:: *)
RGtensors[gbg,x];


(* ::Input::Initialization:: *)
trh=multiDot[gbg,hoddfull,{1,1},{2,2}];


(* ::Input::Initialization:: *)
deltaG=-1/2*covD[covD[trh]]+1/2*(covDiv[Raise[covD[hoddfull],2],{2}]+Transpose[covDiv[Raise[covD[hoddfull],2],{2}]])-1/2*covDiv[Raise[covD[hoddfull],3],{3}]-1/2*gbg*covDiv[covDiv[hoddfull,{{1},2}],{1}]+1/2*gbg*covDiv[Raise[covD[trh],1],{1}]//FullSimplify;


(* ::Input::Initialization:: *)
rule=LegendreP[1+l,Cos[theta]]:>Cos[theta] LegendreP[l,Cos[theta]]-Sphi/(l+1);


(* ::Subsection:: *)
(*Time dependent GR perturbation*)


(* ::Input::Initialization:: *)
hoddfullt={{0,0,0,k0[r,t] Sa[[2]]},{0,0,0, k1[r,t]Sa[[2]]},{0,0,0,0},{ k0[r,t] Sa[[2]], k1[r,t] Sa[[2]],0,0}};


(* ::Input::Initialization:: *)
trh=multiDot[gUU,hoddfullt,{1,1},{2,2}];


(* ::Input::Initialization:: *)
deltaGt=-1/2*covD[covD[trh]]+1/2*(covDiv[Raise[covD[hoddfullt],2],{2}]+Transpose[covDiv[Raise[covD[hoddfullt],2],{2}]])-1/2*covDiv[Raise[covD[hoddfullt],3],{3}]-1/2*gbg*covDiv[covDiv[hoddfullt,{{1},2}],{1}]+1/2*gbg*covDiv[Raise[covD[trh],1],{1}]//FullSimplify;


(* ::Input::Initialization:: *)
deltaGtr=ConstantArray[0,{4,4}];


(* ::Input::Initialization:: *)
ruleSimp={Hypergeometric2F1[-1-l,2+l,1,Sin[theta/2]^2]:>LegendreP[l+1,Cos[theta]],Hypergeometric2F1[-2-l,3+l,1,Sin[theta/2]^2]:>LegendreP[l+2,Cos[theta]],LegendreP[1+l,1,Cos[theta]]:>-(((l+1) (LegendreP[l,Cos[theta]]-Cos[theta] LegendreP[l+1,Cos[theta]]))/Sin[theta])};


(* ::Input::Initialization:: *)
ruleFtoP[expr_]:=expr//.Hypergeometric2F1[a_,b_,1,z_]/;Simplify[b==1-a]:>LegendreP[-a,1-2 z]


(* ::Input::Initialization:: *)
deltaGtr[[1,4]]=FullSimplify[deltaGt[[1,4]]/.{Hypergeometric2F1[-1-l,2+l,1,Sin[theta/2]^2]:>LegendreP[l+1,Cos[theta]],LegendreP[l+1,1,Cos[theta]]:>-((l+1)/Sin[theta])*(LegendreP[l,Cos[theta]]-Cos[theta]*LegendreP[l+1,Cos[theta]])},Assumptions->{Pi>theta>0,l\[Element]Integers&&l>=0,r>0}]/Sa[[2]];


(* ::Input::Initialization:: *)
deltaGtr[[2,4]]=FullSimplify[deltaGt[[2,4]]/.{Hypergeometric2F1[-1-l,2+l,1,Sin[theta/2]^2]:>LegendreP[l+1,Cos[theta]],LegendreP[l+1,1,Cos[theta]]:>-((l+1)/Sin[theta])*(LegendreP[l,Cos[theta]]-Cos[theta]*LegendreP[l+1,Cos[theta]])},Assumptions->{Pi>theta>0,l\[Element]Integers&&l>=0,r>0}]/Sa[[2]];


(* ::Input::Initialization:: *)
deltaGtr[[3,4]]=deltaGt[[3,4]]/Stenab[[1,2]];


(* ::Input::Initialization:: *)
deltaGtr=deltaGtr+Transpose[deltaGtr];


(* ::Input::Initialization:: *)
deltaGAB={deltaGtr[[1,4]],deltaGtr[[2,4]]}


(* ::Input::Initialization:: *)
DStart[f_][r_,t_]:=(1-2M/r)D[f[r,t],r]


(* ::Input::Initialization:: *)
DStar[f_][r_]:=(1-2M/r)D[f[r],r]


(* ::Input::Initialization:: *)
LinearEps[expr_]:=SeriesCoefficient[expr,{\[Epsilon],0,0}]+\[Epsilon]*SeriesCoefficient[expr,{\[Epsilon],0,1}]


(* ::Input::Initialization:: *)
LinearAlpha[expr_]:=SeriesCoefficient[expr,{\[Alpha],0,0}]+\[Alpha]*SeriesCoefficient[expr,{\[Alpha],0,1}]


(* ::Input::Initialization:: *)
normalSol[expr_]:=expr/. ConditionalExpression[x_,cond_]:>x


(* ::Input::Initialization:: *)
VGR[r_]:=(1-2M/r)(l (l+1)/r^2-6M/r^3)


(* ::Input::Initialization:: *)
k0[r_,t_]:=Exp[-I*\[Omega]*t]*k0[r]


(* ::Input::Initialization:: *)
k1[r_,t_]:=Exp[-I*\[Omega]*t]*k1[r]


(* ::Input::Initialization:: *)
solk0GR=Solve[deltaGtr[[3,4]]==0,k0[r]]//FullSimplify;


(* ::Input::Initialization:: *)
solk0d1GR=D[solk0GR,r]//FullSimplify;


(* ::Input::Initialization:: *)
solk0d2GR=D[solk0d1GR,r]//FullSimplify;


(* ::Input::Initialization:: *)
solk1d2GR=Solve[((deltaGtr[[2,4]]/.solk0d1GR[[1]])/.solk0GR[[1]])==0,k1''[r]]//FullSimplify//normalSol;


RGtensors[gtr,xtr];


(* ::Input::Initialization:: *)
Solk1d2RW=Solve[((\[Psi]t[r,t]/.solk0d1GR[[1]])/.solk0GR[[1]])==\[Psi]RW[r]*E^(-I t \[Omega]) ,k1''[r]]//FullSimplify;


(* ::Input::Initialization:: *)
solvek1RW=Solve[((k1''[r]/.solk1d2GR[[1]])-(k1''[r]/.Solk1d2RW[[1]]))==0 && r!= 2M && r!=0 && l^2+l-2!=0,k1[r]]//FullSimplify//normalSol;


(* ::Input::Initialization:: *)
solk1d3GR=D[solk1d2GR,r]//FullSimplify;


solk1d4GR=D[solk1d3GR,r]//FullSimplify;


(* ::Input::Initialization:: *)
solk0d3GR=D[solk0d2GR,r]//FullSimplify;


solk0d4GR=D[solk0d3GR,r]//FullSimplify;


solk1d1RW=D[solvek1RW,r]//FullSimplify;


(* ::Input::Initialization:: *)
solk1d3RW=D[Solk1d2RW,r]//FullSimplify;


(* ::Input::Initialization:: *)
solk1d4RW=D[solk1d3RW,r]//FullSimplify;


(* ::Input::Initialization:: *)
solk1d1RW=D[solvek1RW,r]//FullSimplify;
