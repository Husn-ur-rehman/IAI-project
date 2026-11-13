% =====================================================================
% explain.pl: explanation facility.
%   how(P, Goal, Lines):     how a conclusion was reached (the proof tree as text)
%   why(P, Lines):           why this action was taken (decision + proof of cheating)
%   why_not(P, Goal, Lines): why a conclusion did NOT fire (each rule's weakest premise)
% Lines is a list of strings, so Python and the UI can show it directly.
% =====================================================================

why_not(P, Goal, [Head | Lines]) :-
    prove(P, Goal, CF, _),
    cf_threshold(T),
    format(string(Head), "~w: CF ~2f (rules need every premise CF > ~2f)", [Goal, CF, T]),
    findall(L, ( br(Id, Goal, Prems, RCF), rule_status(P, Id, Prems, RCF, T, L) ), Lines).

rule_status(P, Id, Prems, RCF, T, Line) :-
    findall(S, ( member(Pr, Prems), prove(P, Pr, C, _),
                 format(string(S), "~w=~2f", [Pr, C]) ), Parts),
    findall(C, ( member(Pr, Prems), prove(P, Pr, C, _) ), Cs),
    min_list(Cs, Min),
    atomic_list_concat(Parts, ', ', PartsA),
    (   Min > T
    ->  C1 is Min * RCF,
        format(string(Line), "  rule ~w FIRED (~2f): ~w", [Id, C1, PartsA])
    ;   format(string(Line), "  rule ~w not fired, weakest premise ~2f: ~w", [Id, Min, PartsA])
    ).

how(P, Goal, Lines) :-
    prove(P, Goal, _, Proof),
    phrase(proof_lines(Proof, 0), Lines).

why(P, Lines) :-
    decide(P, Action, CF, Reason),
    action_mode(Action, Mode),
    reason_text(Reason, CF, RText),
    format(string(Head), "Action: ~w (~w) because ~w", [Action, Mode, RText]),
    (   Reason == insufficient_evidence
    ->  Lines = [Head]
    ;   how(P, cheating, Body), Lines = [Head | Body]
    ).

reason_text(insufficient_evidence, _, "there is not enough shot data to judge aim").
reason_text(cheating_cf_at_least(A, T), CF, S) :-
    format(string(S), "cheating CF ~2f >= ~w threshold ~2f", [CF, A, T]).
reason_text(cheating_cf_below_monitor, CF, S) :-
    action_threshold(monitor, T),
    format(string(S), "cheating CF ~2f < monitor threshold ~2f", [CF, T]).

% symptom_labels(P, Labels): short labels of the derived symptoms, e.g. "abnormal(accuracy)@p99"
symptom_labels(P, Labels) :-
    findall(L, ( derived(P, S, _, _, Ev),
                 (   Ev = exceeds(_, _, Level, _)
                 ->  format(string(L), "~w@~w", [S, Level])
                 ;   format(string(L), "~w", [S]) ) ),
            Labels).

% explanation(P, Action, Lines): "why" for flagged players, "why not" for cleared ones
explanation(P, clear, Lines) :- !, why_not(P, cheating, Lines).
explanation(P, _, Lines) :- why(P, Lines).

% ---------- proof tree -> indented lines ---------------------------------

proof_lines(concluded(G, CF, []), D) --> !,
    line(D, "~w: CF ~2f (no rule fired)", [G, CF]).
proof_lines(concluded(G, CF, Rules), D) -->
    line(D, "~w: CF ~2f", [G, CF]),
    { D1 is D + 1 },
    rules_lines(Rules, D1).
proof_lines(fact(S, CF, Id, Ev), D) -->
    { evidence_text(Ev, ET) },
    line(D, "~w: CF ~2f, ~w [~w]", [S, CF, ET, Id]).
proof_lines(not(G, CF0, _), D) -->
    line(D, "not ~w (its CF ~2f is at or below the threshold)", [G, CF0]).
proof_lines(absent(S), D) -->
    line(D, "~w: not observed", [S]).

rules_lines([], _) --> [].
rules_lines([rule(Id, RCF, C, Subs) | Rs], D) -->
    { ( RCF < 0 -> Dir = "against" ; Dir = "for" ) },
    line(D, "rule ~w (~w, rule CF ~2f) contributes ~2f", [Id, Dir, RCF, C]),
    { D1 is D + 1 },
    subs_lines(Subs, D1),
    rules_lines(Rs, D).

subs_lines([], _) --> [].
subs_lines([S | Ss], D) --> proof_lines(S, D), subs_lines(Ss, D).

line(D, Fmt, Args) -->
    { Pad is D * 2, format(string(Body), Fmt, Args),
      format(string(L), "~*c~w", [Pad, 0' , Body]) },
    [L].

evidence_text(exceeds(M, V, Level, T), S) :- !,
    format(string(S), "~w = ~3f > ~w (~3f)", [M, V, Level, T]).
evidence_text(strongest_of(Ms), S) :- !,
    atomic_list_concat(Ms, ', ', A), format(string(S), "abnormal impact: ~w", [A]).
evidence_text(measured_and_normal(Ms), S) :- !,
    atomic_list_concat(Ms, ', ', A), format(string(S), "~w measured and within normal range", [A]).
evidence_text(no_mechanical_metric(R), S) :- !,
    format(string(S), "no aim metric has enough data (~w rounds)", [R]).
evidence_text(E, S) :- format(string(S), "~w", [E]).
