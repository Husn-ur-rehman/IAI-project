% Knowledge base entry point.
% Layers (filled in step 4):
%   1. stats    -> symptoms   (forward inference from asserted player facts)
%   2. symptoms -> diagnosis  (cheat type + certainty factor)
%   3. diagnosis -> action    (automatic decision or human review)
%   + why/how explanation meta-interpreter

:- dynamic stat/3.          % stat(PlayerMatchId, Feature, Value)
:- dynamic threshold/3.     % threshold(Feature, Percentile, Value)

kb_ready.
