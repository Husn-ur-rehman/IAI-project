% =====================================================================
% symptoms.pl: LAYER 1 knowledge, metrics -> symptoms (forward chaining).
% fr(Stratum, RuleId, P, Symptom, CF, Evidence, Body)
% =====================================================================

% Severity grading against thresholds calibrated on clean matches.
% Above p99 is strong evidence and above p95 is weak evidence.
severity(p99, 0.8).
severity(p95, 0.4).

grade(M, V, CF, Level, T) :-
    (   threshold(M, p99, T), V > T -> Level = p99
    ;   threshold(M, p95, T), V > T -> Level = p95
    ),
    severity(Level, CF).

% Mechanical metrics measure aim. Impact metrics measure results.
mechanical_metric(headshot_rate).
mechanical_metric(accuracy).
mechanical_metric(hs_kill_rate).
impact_metric(adr).
impact_metric(kpr).
impact_metric(kd).

% ---- Stratum 1: symptoms read directly from metric values -----------

% S1: a metric exceeds its calibrated percentile.
fr(1, s_abnormal, P, abnormal(M), CF, exceeds(M, V, Level, T),
   ( value(P, M, V), grade(M, V, CF, Level, T) )).

% S2: there are too few shots or hits to judge aim at all.
fr(1, s_insufficient, P, insufficient_evidence, 1.0, no_mechanical_metric(Rounds),
   ( \+ ( mechanical_metric(M), has_evidence(P, M) ),
     count(P, rounds_played, Rounds) )).

% ---- Stratum 2: symptoms built on stratum-1 symptoms ----------------

% S3: high impact. The strongest abnormal impact metric gives the CF.
fr(2, s_high_impact, P, high_impact, CF, strongest_of(Ms),
   ( findall(M-C, ( impact_metric(M), symptom(P, abnormal(M), C) ), Pairs),
     Pairs \== [],
     pairs_keys_values(Pairs, Ms, Cs), max_list(Cs, CF) )).

% S4: normal mechanics. Aim was measurable and nothing about it was abnormal.
%     Uses negation of stratum-1 symptoms, which is why it is in stratum 2.
fr(2, s_normal_mechanics, P, normal_mechanics, 0.8, measured_and_normal([headshot_rate, accuracy]),
   ( has_evidence(P, headshot_rate), has_evidence(P, accuracy),
     \+ ( mechanical_metric(M), symptom(P, abnormal(M), _) ) )).
