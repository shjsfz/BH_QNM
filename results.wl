(* ::Package:: *)

(* ::Text:: *)
(*For vacuum R3 theory*)


(* ::Input::Initialization:: *)
VGR[r_]:=(1-2M/r)(l (l+1)/r^2-6M/r^3)


VmodR3[r_]:=VGR[r]-\[Alpha]*1/(4 M^2 r^10) (-410880 M^6-16 (-34609+1754 l (1+l)) M^5 r+8 M^4 r^2 (-30469+3204 l (1+l)-376 r^2 \[Omega]^2)+10 M r^5 (5+2 r^2 \[Omega]^2)+5 r^6 (3-2 l (1+l)+2 r^2 \[Omega]^2)+4 M^3 r^3 (8645-1440 l (1+l)+308 r^2 \[Omega]^2)+10 M^2 (r^4+4 r^6 \[Omega]^2))


AR3[r_]:=(r^6 (-2 M+r)+M (648 M^2-642 M r+144 r^2+(5 r^5)/(8 M^3)) \[Alpha])/(r^4 (-2 M+r)^2)


(* ::Text:: *)
(*k1[r] = AR3[r,\[Omega]]*(\[Psi]RW'). k1 GR is the [2,4] component of the perturbation*)
(*\[Psi]RW' satisfies the modified RW eq:*)
(*(d^2/SubStar[dr]^2 + \[Omega]^2 - VmodR3) \[Psi]RW' =0, here SubStar[r] is the GR tortoise coordinate*)


(* ::Text:: *)
(*For RK theory*)


(* ::Input::Initialization:: *)
ARK[r_]:=(-2 M r^6+r^7+336 M^3 \[Alpha]-204 M^2 r \[Alpha]+(3 r^5 \[Alpha])/(4 M^2))/(r^4 (-2 M+r)^2)


(* ::Input::Initialization:: *)
VmodRK[r_]:=VGR[r]-\[Alpha]*1/(2 M^2 r^10) 3 (-3840 M^6-16 (-317+22 l (1+l)) M^5 r+8 M^4 r^2 (-209+24 l (1+l)-16 r^2 \[Omega]^2)+2 M r^5 (5+2 r^2 \[Omega]^2)+r^6 (3-2 l (1+l)+2 r^2 \[Omega]^2)+4 M^3 (r^3+4 r^5 \[Omega]^2)+2 M^2 (r^4+4 r^6 \[Omega]^2))
