% =====================================================================
% actions.pl: LAYER 3 knowledge, diagnosis -> action.
%
% DECISION MAKING: the system acts on its own only when the action is
%   cheap to reverse (clear, monitor, hide from the leaderboard).
% DECISION SUPPORT: a ban is never automatic. Above the review threshold the
%   player is hidden and queued for a human, with the evidence attached.
% =====================================================================

:- dynamic action_threshold/2.     % the UI can change these at runtime

% action_threshold(Action, MinCheatingCF): checked from the top down
action_threshold(review_for_ban,        0.75).
action_threshold(hide_from_leaderboard, 0.5).
action_threshold(monitor,               0.2).

action_mode(no_action,             automatic).
action_mode(clear,                 automatic).
action_mode(monitor,               automatic).
action_mode(hide_from_leaderboard, automatic).
action_mode(review_for_ban,        human).        % decision support only

% decide(P, Action, CF, Reason)
decide(P, no_action, 0.0, insufficient_evidence) :-
    symptom(P, insufficient_evidence, _), !.
decide(P, Action, CF, cheating_cf_at_least(Action, T)) :-
    prove(P, cheating, CF, _),
    action_threshold(Action, T),
    CF >= T, !.
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
