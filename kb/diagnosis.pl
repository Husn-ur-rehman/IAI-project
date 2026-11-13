% =====================================================================
% diagnosis.pl: LAYER 2 knowledge, symptoms -> diagnoses (backward chaining, CF).
% br(RuleId, Conclusion, Premises, RuleCF)
%
% The rule CFs are knowledge-engineering judgements, set BEFORE any evaluation
% against labels. They are not fitted to the data.
% =====================================================================

% ---- Aimbot: aim that is unnaturally precise --------------------------
br(a1, aimbot, [abnormal(headshot_rate), abnormal(accuracy)],     0.85).
br(a2, aimbot, [abnormal(headshot_rate), abnormal(hs_kill_rate)], 0.6).
br(a3, aimbot, [abnormal(accuracy), high_impact],                 0.5).
br(a4, aimbot, [abnormal(headshot_rate), high_impact],            0.5).

% ---- Wallhack: knowing where enemies are behind cover -----------------
br(w1, wallhack, [abnormal(wallbang_rate)],                  0.5).
br(w2, wallhack, [abnormal(wallbang_rate), high_impact],     0.7).
br(w3, wallhack, [abnormal(smoke_kill_rate), high_impact],   0.3).

% ---- Rage cheat: blatant cheating, e.g. killing while airborne or flashed
br(r1, rage_cheat, [abnormal(air_kill_rate), abnormal(headshot_rate)],   0.7).
br(r2, rage_cheat, [abnormal(blind_kill_rate), abnormal(headshot_rate)], 0.5).
br(r3, rage_cheat, [abnormal(wallbang_rate), abnormal(headshot_rate),
                    abnormal(accuracy)],                                 0.8).

% ---- Legitimate outlier: a strong player whose aim is normal ----------
br(l1, legit_outlier, [high_impact, normal_mechanics], 0.8).

% ---- Overall verdict ----------------------------------------------------
br(c1, cheating, [aimbot],        1.0).
br(c2, cheating, [wallhack],      1.0).
br(c3, cheating, [rage_cheat],    1.0).
br(c4, cheating, [legit_outlier], -0.7).   % evidence AGAINST cheating

cheat_type(aimbot).
cheat_type(wallhack).
cheat_type(rage_cheat).

% primary_diagnosis(P, Type, CF): the cheat type with the highest CF, or none.
primary_diagnosis(P, Type, CF) :-
    findall(C-T, ( cheat_type(T), prove(P, T, C, _) ), Pairs),
    max_member(Best-BestT, Pairs),
    cf_threshold(Th),
    ( Best > Th -> Type = BestT, CF = Best ; Type = none, CF = 0.0 ).
