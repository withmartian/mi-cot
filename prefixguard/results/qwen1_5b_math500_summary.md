# SDS/CEBRA Prefix-Verifier Analysis

## Core Selection Accuracy
- `first`: 0.117
- `random`: 0.133
- `majority_answer`: 0.150
- `longest`: 0.117
- `shortest`: 0.200
- `avg_lp`: 0.217
- `state_score`: 0.133
- `shuffled_state_score`: 0.150
- `cebra_score`: 0.250
- `joint_state`: 0.183
- `joint_cebra`: 0.250
- `transition_ll`: 0.217
- `oracle`: 0.383

## Information Gap
- `logprob`: candidate AUC 0.593, pick accuracy 0.217
- `cebra`: candidate AUC 0.748, pick accuracy 0.250
- `logprob_plus_cebra`: candidate AUC 0.743, pick accuracy 0.267
- `logprob_plus_state_transition`: candidate AUC 0.722, pick accuracy 0.200
- `all_sds`: candidate AUC 0.759, pick accuracy 0.250
- `all`: candidate AUC 0.753, pick accuracy 0.250
- `corr(avg_lp, cebra_score)`: 0.270

## Robustness Slices
- `all` n=60 oracle=0.383: avg_lp=0.217, cebra=0.250, joint=0.250
- `answer_form:fraction` n=12 oracle=0.167: avg_lp=0.083, cebra=0.083, joint=0.083
- `answer_form:integer` n=37 oracle=0.405: avg_lp=0.243, cebra=0.243, joint=0.243
- `answer_form:symbolic` n=10 oracle=0.500: avg_lp=0.200, cebra=0.400, joint=0.400
- `answer_len:long_answer` n=19 oracle=0.211: avg_lp=0.053, cebra=0.105, joint=0.105
- `answer_len:mid_answer` n=13 oracle=0.462: avg_lp=0.308, cebra=0.385, joint=0.385
- `answer_len:short_answer` n=28 oracle=0.464: avg_lp=0.286, cebra=0.286, joint=0.286
- `oracle:False` n=37 oracle=0.000: avg_lp=0.000, cebra=0.000, joint=0.000
- `oracle:True` n=23 oracle=1.000: avg_lp=0.565, cebra=0.652, joint=0.652
- `problem_len:long` n=19 oracle=0.263: avg_lp=0.105, cebra=0.158, joint=0.158
- `problem_len:middle` n=21 oracle=0.333: avg_lp=0.190, cebra=0.143, joint=0.143
- `problem_len:short` n=20 oracle=0.550: avg_lp=0.350, cebra=0.450, joint=0.450

## Early Exit
- `0.25`: avg_lp=0.133, cebra=0.167, state=0.167
- `0.5`: avg_lp=0.167, cebra=0.183, state=0.167
- `0.75`: avg_lp=0.183, cebra=0.183, state=0.133
- `1.0`: avg_lp=0.217, cebra=0.250, state=0.133

## Interpretability Example
- Problem id: `20`
- Answer: `6+9i`
- SDS score that fixed log-prob: `cebra_score`
- Log-prob pick: correct=False, avg_lp=-0.343, cebra=0.328, detour=0.092
- SDS pick: correct=True, avg_lp=-0.433, cebra=0.380, detour=0.046
- Log-prob state summary:
```json
{
  "n_states": 65,
  "mean_state_utility": 0.6695731058762945,
  "risk_state_frac": 0.09230769230769231,
  "state_histogram": {
    "1": 50,
    "5": 6,
    "0": 9
  },
  "state_rle": [
    [
      1,
      3
    ],
    [
      5,
      2
    ],
    [
      1,
      5
    ],
    [
      5,
      1
    ],
    [
      1,
      1
    ],
    [
      5,
      2
    ],
    [
      1,
      3
    ],
    [
      5,
      1
    ],
    [
      1,
      13
    ],
    [
      0,
      2
    ],
    [
      1,
      3
    ],
    [
      0,
      1
    ],
    [
      1,
      9
    ],
    [
      0,
      2
    ],
    [
      1,
      2
    ],
    [
      0,
      1
    ],
    [
      1,
      7
    ],
    [
      0,
      1
    ],
    [
      1,
      1
    ],
    [
      0,
      2
    ],
    [
      1,
      3
    ]
  ]
}
```
- SDS state summary:
```json
{
  "n_states": 65,
  "mean_state_utility": 0.7128576020155791,
  "risk_state_frac": 0.046153846153846156,
  "state_histogram": {
    "1": 52,
    "5": 2,
    "4": 1,
    "0": 10
  },
  "state_rle": [
    [
      1,
      5
    ],
    [
      5,
      1
    ],
    [
      1,
      9
    ],
    [
      4,
      1
    ],
    [
      1,
      2
    ],
    [
      0,
      1
    ],
    [
      1,
      1
    ],
    [
      0,
      1
    ],
    [
      5,
      1
    ],
    [
      1,
      17
    ],
    [
      0,
      2
    ],
    [
      1,
      8
    ],
    [
      0,
      2
    ],
    [
      1,
      1
    ],
    [
      0,
      2
    ],
    [
      1,
      2
    ],
    [
      0,
      1
    ],
    [
      1,
      7
    ],
    [
      0,
      1
    ]
  ]
}
```
