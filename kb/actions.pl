% =====================================================================
% actions.pl: LAYER 3 knowledge, diagnosis -> action.
%
% DECISION MAKING: the system acts on its own only when the action is
%   cheap to reverse (clear, monitor, hide from the leaderboard).
% DECISION SUPPORT: a ban is never automatic. Above the review threshold the
%   player is hidden and queued for a human, with the evidence attached.
% =====================================================================

% default_action_threshold(Action, MinCheatingCF)
default_action_threshold(review_for_ban,        0.75).
default_action_threshold(hide_from_leaderboard, 0.5).
default_action_threshold(monitor,               0.2).

% action_threshold/2 holds the thresholds in force; the strictness dial (UI / sweep) changes them.
:- dynamic action_threshold/2.

reset_action_thresholds :-
    retractall(action_threshold(_, _)),
    forall(default_action_threshold(A, T), assertz(action_threshold(A, T))).

set_action_threshold(A, T) :-
    default_action_threshold(A, _),            % only known actions
    retractall(action_threshold(A, _)),
    assertz(action_threshold(A, T)).

:- reset_action_thresholds.

action_mode(no_action,             automatic).
action_mode(clear,                 automatic).
action_mode(monitor,               automatic).
action_mode(hide_from_leaderboard, automatic).
action_mode(review_for_ban,        human).        % decision support only

% decide(P, Action, CF, Reason)
decide(P, no_action, 0.0, insufficient_evidence) :-
    symptom(P, insufficient_evidence, _), !.
% The strictest action whose threshold the CF reaches (does not depend on clause order).
decide(P, Action, CF, cheating_cf_at_least(Action, T)) :-
    prove(P, cheating, CF, _),
    findall(T0-A0, ( action_threshold(A0, T0), CF >= T0 ), Reached),
    Reached \== [],
    max_member(T-Action, Reached), !.
decide(P, clear, CF, cheating_cf_below_monitor) :-
    prove(P, cheating, CF, _).

% assess(P, Action, Mode, CheatingCF, Diagnosis, DiagnosisCF)
% The single entry point for Python. It runs forward chaining, then backward chaining.
assess(P, Action, Mode, CF, Diagnosis, DCF) :-
    forward(P),
    decide(P, Action, CF0, _),
    action_mode(Action, Mode),
    CF is max(CF0, 0.0),           % a negative CF means evidence of legit play; show it as 0
    primary_diagnosis(P, Diagnosis, DCF).
