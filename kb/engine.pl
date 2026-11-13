% =====================================================================
% engine.pl: domain-independent inference engine.
% It knows nothing about CS2; all domain knowledge is in symptoms.pl / diagnosis.pl.
%
%  FORWARD CHAINING (data-driven), used for layer 1 (facts -> symptoms)
%    fr(Stratum, RuleId, P, Symptom, CF, Evidence, Body)
%    Rules fire to a fixpoint, one stratum at a time. A stratum may only
%    negate symptoms from lower strata (stratified negation).
%    Results: derived(P, Symptom, CF, RuleId, Evidence)
%
%  BACKWARD CHAINING (goal-driven) with MYCIN certainty factors, used for layer 2
%    br(RuleId, Conclusion, Premises, RuleCF)
%    Premise: a symptom | a conclusion of other br/4 rules | not(Premise)
%    CF(premises) = min(CF of each premise); the rule fires if that is > 0.2.
%    CF(rule)     = CF(premises) * RuleCF  (RuleCF may be negative: evidence against)
%    Rules that reach the same conclusion are merged with MYCIN's combination function.
% =====================================================================

:- dynamic derived/5.

cf_threshold(0.2).

% ---------- Forward chaining ------------------------------------------

forward(P) :-
    retractall(derived(P, _, _, _, _)),
    findall(S, fr(S, _, _, _, _, _, _), Ss), sort(Ss, Strata),
    forall(member(S, Strata), saturate(P, S)).

saturate(P, S) :-
    (   fr(S, Id, P, Sym, CF, Ev, Body),
        call(Body),
        \+ derived(P, Sym, _, _, _)
    ->  assertz(derived(P, Sym, CF, Id, Ev)),
        saturate(P, S)
    ;   true
    ).

symptom(P, Sym, CF) :- derived(P, Sym, CF, _, _).

% ---------- Backward chaining with certainty factors ------------------

goal(G) :- br(_, G, _, _), !.

% prove(+P, +Goal, -CF, -Proof)
prove(P, not(G), CF, not(G, CF0, Sub)) :- !,
    prove(P, G, CF0, Sub),
    cf_threshold(T),
    ( CF0 =< T -> CF = 1.0 ; CF = 0.0 ).
prove(P, G, CF, Proof) :-
    goal(G), !,
    findall(C-rule(Id, RuleCF, C, Subs),
            ( br(Id, G, Premises, RuleCF),
              fire(P, Premises, RuleCF, C, Subs) ),
            Fired),
    pairs_keys(Fired, Cs),
    foldl(combine, Cs, 0.0, CF),
    pairs_values(Fired, Rules),
    Proof = concluded(G, CF, Rules).
prove(P, Sym, CF, fact(Sym, CF, Id, Ev)) :-
    derived(P, Sym, CF, Id, Ev), !.
prove(_, Sym, 0.0, absent(Sym)).

fire(P, Premises, RuleCF, C, Subs) :-
    maplist(prove_premise(P), Premises, CFs, Subs),
    min_list(CFs, Min),
    cf_threshold(T),
    Min > T,
    C is Min * RuleCF.

prove_premise(P, Prem, CF, Proof) :- prove(P, Prem, CF, Proof).

% MYCIN combination of two certainty factors
combine(X, Y, Z) :- X >= 0, Y >= 0, !, Z is X + Y * (1 - X).
combine(X, Y, Z) :- X <  0, Y <  0, !, Z is X + Y * (1 + X).
combine(X, Y, Z) :- Z is (X + Y) / (1 - min(abs(X), abs(Y))).
