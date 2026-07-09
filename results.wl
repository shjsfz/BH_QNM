(* ::Package:: *)

(* ::Text:: *)
(*For vacuum R3 theory*)


(* ::Input::Initialization:: *)
VGR[r_]:=(1-2M/r)(l (l+1)/r^2-6M/r^3)


VmodR3[r_]:=VGR[r]+(2 M-r)1/((-2+l+l^2) r^11) 24 M \[Alpha] (-34804 M^3-4 (-13091+263 l (1+l)) M^2 r-5 (-2+l) (3+l) (156+7 l (1+l)) r^3-8 (-87+l+l^2) r^5 \[Omega]^2+M r^2 (l (1+l) (1727+83 l (1+l))-6 (4489+160 r^2 \[Omega]^2)))


AR3[r_]:=1-(12 M \[Alpha] (1468 M^2+2 (113+96 l (1+l)) M r+3 r^2 (-132-29 l (1+l)+8 r^2 \[Omega]^2)))/((-2+l+l^2) r^7)


(* ::Text:: *)
(*\[Psi]RW_{GR} = AR3[r,\[Omega]]*\[Psi]RW'. Here \[Psi]RW_{GR} has exactly the same definition as in GR*)
(*\[Psi]RW' satisfies the modified RW eq:*)
(*(d^2/SubStar[dr]^2 + \[Omega]^2 - VmodR3) \[Psi]RW' =0, here SubStar[r] is the GR tortoise coordinate*)


(* ::Text:: *)
(*For RK theory*)


(* ::Input::Initialization:: *)
ARK[r_]:=1-(24 M^2 (-108 M+(52+l+l^2) r) \[Alpha])/((-2+l+l^2) r^7)


(* ::Input::Initialization:: *)
VmodRK[r_]:=VGR[r]+\[Alpha] (144 M^2 (2 M-r) (3388 M^2-4 (647+4 l (1+l)) M r+9 (52+l+l^2) r^2-20 r^4 \[Omega]^2))/((-2+l+l^2) r^11)
