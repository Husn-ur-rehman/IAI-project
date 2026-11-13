% =====================================================================
% metrics.pl: the single definition of every metric the KB reasons over.
%
% Base facts are produced by Python (cheatdetect.facts), loaded with load_facts/1:
%   stat(Pid, Key, Count)       counts per player-match
%   kill(Pid, Weapon, Headshot, ThruSmoke, Wallbang, AttackerBlind,
%        NoScope, InAir, Distance)   one per kill, booleans are true/false
% =====================================================================

:- dynamic stat/3.
:- dynamic kill/9.

% ---------- Loading base facts ----------------------------------------

load_facts(File) :-
    retractall(stat(_, _, _)),
    retractall(kill(_, _, _, _, _, _, _, _, _)),
    setup_call_cleanup(open(File, read, S), read_facts(S), close(S)).

read_facts(S) :-
    read_term(S, T, []),
    (   T == end_of_file -> true
    ;   assertz(T), read_facts(S)
    ).

player(P) :- stat(P, rounds_played, _).

% ---------- Base counts -----------------------------------------------

count(P, Key, N) :- stat(P, Key, N), !.
count(_, _, 0).

kill_count(P, all,       N) :- aggregate_all(count, kill(P,_,_,_,_,_,_,_,_), N).
kill_count(P, headshot,  N) :- aggregate_all(count, kill(P,_,true,_,_,_,_,_,_), N).
kill_count(P, thrusmoke, N) :- aggregate_all(count, kill(P,_,_,true,_,_,_,_,_), N).
kill_count(P, wallbang,  N) :- aggregate_all(count, kill(P,_,_,_,true,_,_,_,_), N).
kill_count(P, blind,     N) :- aggregate_all(count, kill(P,_,_,_,_,true,_,_,_), N).
kill_count(P, in_air,    N) :- aggregate_all(count, kill(P,_,_,_,_,_,_,true,_), N).

% ---------- Metric definitions ----------------------------------------
% metric(Name, Description)
metric(headshot_rate,   'share of gun hits that hit the head').
metric(accuracy,        'share of gun shots that hit an enemy').
metric(adr,             'average damage per round').
metric(kpr,             'kills per round').
metric(kd,              'kills per death').
metric(hs_kill_rate,    'share of kills that are headshots').
metric(wallbang_rate,   'share of kills through walls').
metric(smoke_kill_rate, 'share of kills through smoke').
metric(blind_kill_rate, 'share of kills while flashed').
metric(air_kill_rate,   'share of kills while airborne').

% raw_value(P, Metric, Value): the arithmetic definition
raw_value(P, headshot_rate, V)   :- ratio(P, head_hits, gun_hits, V).
raw_value(P, accuracy, V)        :- ratio(P, gun_hits, shots_fired, V).
raw_value(P, adr, V)             :- ratio(P, damage, rounds_played, V).
raw_value(P, kpr, V)             :- ratio(P, kills, rounds_played, V).
raw_value(P, kd, V)              :- count(P, kills, K), count(P, deaths, D), V is K / max(D, 1).
raw_value(P, hs_kill_rate, V)    :- kill_ratio(P, headshot, V).
raw_value(P, wallbang_rate, V)   :- kill_ratio(P, wallbang, V).
raw_value(P, smoke_kill_rate, V) :- kill_ratio(P, thrusmoke, V).
raw_value(P, blind_kill_rate, V) :- kill_ratio(P, blind, V).
raw_value(P, air_kill_rate, V)   :- kill_ratio(P, in_air, V).

ratio(P, Num, Den, V) :- count(P, Num, N), count(P, Den, D), D > 0, V is N / D.
kill_ratio(P, Kind, V) :- kill_count(P, Kind, N), kill_count(P, all, D), D > 0, V is N / D.

% ---------- Evidence requirements --------------------------------------
% A rate is only meaningful with a large enough denominator.
% min_evidence(Metric, BaseCount, Minimum)
min_evidence(headshot_rate,   gun_hits,      20).
min_evidence(accuracy,        shots_fired,   50).
min_evidence(adr,             rounds_played,  8).
min_evidence(kpr,             rounds_played,  8).
min_evidence(kd,              rounds_played,  8).
min_evidence(hs_kill_rate,    kills,          8).
min_evidence(wallbang_rate,   kills,          8).
min_evidence(smoke_kill_rate, kills,          8).
min_evidence(blind_kill_rate, kills,          8).
min_evidence(air_kill_rate,   kills,          8).

has_evidence(P, M) :- min_evidence(M, Base, Min), count(P, Base, N), N >= Min.

% value(P, M, V): metric value, defined only when there is enough evidence.
% count/3 commits with a cut, so P must be bound first: enumerate players.
value(P, M, V) :- player(P), metric(M, _), has_evidence(P, M), raw_value(P, M, V).
