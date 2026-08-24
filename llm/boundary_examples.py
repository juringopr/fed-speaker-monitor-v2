# fed_speaker_monitor_v2/llm/boundary_examples.py


BOUNDARY_EXAMPLES = """
============================================================
CLASSIFICATION BOUNDARY EXAMPLES
============================================================

The following examples illustrate important classification
boundaries.

Use them to calibrate the relationship between speaker attribution,
policy relevance, and policy stance.

Do not copy labels mechanically.
Apply the same classification principles to the new input.


------------------------------------------------------------
EXAMPLE 1 — TARGET SPEAKER IS ONLY MENTIONED
------------------------------------------------------------

Target speaker: Kevin Warsh

Text:
"With Kevin Warsh heading the Federal Reserve, investors have
to accept the reality of a new market regime," said a strategist.

Correct interpretation:

policy_relevance_score = 3
policy_relevant = true

speaker_evidence_type = CONTEXT_ONLY
content_type = IRRELEVANT
stance = IRRELEVANT
policy_action = NOT_APPLICABLE
policy_bearing_phrase = ""
stance_driver = NOT_APPLICABLE
text_sufficiency = INSUFFICIENT

Reason:
The segment concerns monetary-policy context and therefore may
be policy-relevant at the segment level.

However, no policy view is attributable to the TARGET SPEAKER.
Therefore it must not contribute to the target speaker's stance.

------------------------------------------------------------
EXAMPLE 2 — DIRECT ECONOMIC DESCRIPTION WITHOUT POLICY DIRECTION
------------------------------------------------------------

Target speaker:
Federal Reserve official

Text:
"The labor market has cooled considerably."

Correct interpretation:

speaker_evidence_type = DIRECT_QUOTE
policy_relevance_score = 3
policy_relevant = true
content_type = IRRELEVANT
stance = IRRELEVANT
policy_action = NOT_APPLICABLE
stance_driver = LABOR
bis_stance = NOT_APPLICABLE
policy_bearing_phrase = ""
text_sufficiency = PARTIAL

Reason:

The target speaker directly describes a macroeconomic condition
that is relevant to monetary policy.

However, the statement does not express a preference for tighter,
easier, or unchanged monetary policy.

The economic description should therefore be preserved as
policy-relevant contextual information, including its LABOR
driver, but it should not contribute to the speaker's hawkish
or dovish stance.

A description of weaker or stronger economic conditions alone
must not be converted into a monetary-policy direction.

------------------------------------------------------------
EXAMPLE 3 — EXPLICIT POLICY-BEARING STATEMENT
------------------------------------------------------------

Target speaker:
Federal Reserve official

Text:
"Inflation remains too high, and we should maintain restrictive
rates until price stability is restored."

Correct interpretation:

speaker_evidence_type = DIRECT_QUOTE
policy_relevant = true
content_type = POLICY_RELEVANT
stance = HAWKISH
policy_action = HOLD_RESTRICTIVE
stance_driver = INFLATION

Reason:

The target speaker explicitly connects elevated inflation to a
preference for maintaining restrictive monetary policy.

This is a policy-bearing statement rather than economic
description alone.


============================================================
BOUNDARY PRINCIPLE
============================================================

Distinguish carefully between:

1. discussion ABOUT the target speaker,
2. economic description BY the target speaker,
3. policy direction expressed BY the target speaker.

A policy-related topic alone does not establish target-speaker
policy relevance.

A policy-relevant economic description does not automatically
establish HAWKISH or DOVISH direction.

Clear policy direction requires policy-bearing language
attributable to the target speaker.
"""