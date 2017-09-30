COMS W4705 PSET #1
------------------

- Hang Gao (uni: 2469)
- Email: [hang.gao@columbia.edu](mailto:hang.gao@columbia.edu)
- Date: Sep 24, 2017


Mannual
-------

* Question 4
* Question 5
* Question 6
* Original Setup
* Advices


QUESTION 4
----------

1. Setup:

    > python HW4_1.py

    **** With total running time of 0.48s

    > python HW4_2.py

    **** With total running time of 1.79s

2. Result:

* 4_1.txt: the merged training data file with new counts, by replacing 
infrequent words with `_RARE_`.

* 4_2.txt: the predictions based on naive HMM which tags only based on 
choosing tags to maximize emissions.

3. Performance

    > cat log/q4.log

    Found 14043 NEs. Expected 5931 NEs; Correct: 3117.

             precision      recall          F1-Score
    Total:   0.221961       0.525544        0.312106
    PER:     0.435451       0.231230        0.302061
    ORG:     0.475936       0.399103        0.434146
    LOC:     0.147750       0.870229        0.252612
    MISC:    0.491689       0.610206        0.544574

4. Observations

* At this section, we trained a baseline model by choosing the tag to 
maximize emission parameters alone for the NER problem. Not Surprisingly, 
the performance is relative weak - at an average F1-Score of `0.31`.

* From a high level, the recall is much higher than precision, indicating 
that the model does relatively well in retrieving the most of positive tags, 
but at the same time producing a lot of false alarms.

* Note that `LOC` category has the lowest precision but highest recall, 
which finally produces the lowest F1-Score. It can be interpreted as the 
false alarms are particularly severe in this tagging task, which is perfectly 
reasonable since the location and person's name can be hard to detach 
without the context information.


QUESTION 5
----------

1. Setup:

    > python HW5_1.py

    **** With total running time of 1.62s

    > python HW5_2.py

    **** With total running time of 17.79s

2. Result:

* trigrams.txt: the enumerated trigrams given the tag set in training data.

* 5_1.txt: the trigrams file with its log probability in training data.

* 5_2.txt: the predictions based on trigram HMM which tags words by MLE on 
trigram transitions and according emissions.

3. Performance

    > cat log/q5.log

    Found 4704 NEs. Expected 5931 NEs; Correct: 3640.

             precision      recall          F1-Score
    Total:   0.773810       0.613724        0.684532
    PER:     0.757660       0.591948        0.664630
    ORG:     0.611855       0.478326        0.536913
    LOC:     0.876458       0.696292        0.776056
    MISC:    0.830065       0.689468        0.753262

4. Observations

* First note that the most improvement is, we now have much less possitive
alerts of NEs (14043/4704) so that our precision is much higher than the 
baseline. This improvement, in my opinion, is close related to the trigram 
MLE and HMM's context information. Since in the baseline model, we cannot 
access to the contextual knowledge, so that we will draw same prediction
for a given word since their probability is fixed. But now, we are not 
constrained by local guessing such that a better understanding of tagging 
could be formed.


QUESTION 6
----------

0. New rule:

To improve HMM performance, I here propose a new set of rules to parse 
infrequent words into different dummy holders:

    (1) `_NUM_`: general numbers, e.g. 4, 1,800, .63 ...

    (2) `_CAP_PARTIAL_`: title-alike words with only the first letter to 
    be captital, e.g. Shi-Ting, New York, Allen ...

    (3) `_CAP_TOTAL_`: all-capital words, e.g. COLUMBIA, X-Y-Z ...

    (4) `_TIME_`: time-alike words, note it should be divided from `_NUN_`, 
    e.g. 12/42-12, 18:00 ...


1. Setup:

    > python HW6.py

    **** With total running time of 18.54s

2. Result:

    * 6_0.txt: the intermediate file by replacing infrequent file by the 
    scheme listed above.

    * 6.txt: the new prediction file by the exact trigram model as in 
    question 5, but with new parsing rule.

3. Performance

    > cat log/q6.log

    Found 5810 NEs. Expected 5931 NEs; Correct: 4331.

             precision      recall          F1-Score
    Total:   0.745439       0.730231        0.737757
    PER:     0.809199       0.775299        0.791887
    ORG:     0.543083       0.668909        0.599464
    LOC:     0.841945       0.755180        0.796206
    MISC:    0.828042       0.679696        0.746571

4. Observations

    * The new paring rules are proved to work: with ~5% and ~12% improvement 
    on F1-Score and recall, at the cost of ~3% precision loss.

    * I found this experiment to be the most interesting. We now know that a
    fine-grained parsing rule could result in performance improvement, even 
    by gap. For this specific task to tag name entities, it is intuition that 
    capital pattern should be taken into account. The help of numbers and 
    times is more subtle: they help the model to work better by making more
    sense of trigram probability. For an example, `I arrive at Columbia on 
    08/26`, in this phrase, timestamp could be helpful by giving more contextual 
    information. At the test phrase, if we have `Charlie goes to Columbia in 
    09/2017`, the model is more likely to find out the location by pointing at 
    time or date, even they are not identical.

    * This a triumph of "handcraft feature", or "expert advice". But it takes 
    time to figure which is better and which is not. For more sophisticated 
    cases, it's probably impossible to enumerate all patterns by human. What 
    if we could discover such representation by machine rather man? By 
    another word, it's more ideal to use Seq2seq encoder for us to find 
    a better feature space at the letter level rather than word level.


ORIGAINAL SETUP
---------------

The code submission instruction comes later than I finished all my design and 
coding. So at the time, I have already build a running script to estimate 
performance for different models on the fly.

Although it might be unnecessary, I still would like to post it here and show 
you there is more efficient way to run question tasks:

    > chmod +x ./run
    > ./run q{4, 5, 6} # For Question 4 / 5 / 6

For an instance, we can now run Question 5 by this script, and get:

    **** With total running time of 16.32s
    Done. Please check log/q5.log for evaluation.

Turns out to be a bit faster than reading and loading from file streams, since 
the most of work is done in the cache.

For more informations, please refer to the code which is well documented.


ADVICE (https://piazza.com/class/j5zcbvz4ofr2wm?cid=33)
------

I have posted this on the piazza, but still would like to reaffirm my advices:

1. a better homework structure: in current instruction, the files are in
the same directory without any helper folders, it's getting really messy. Can 
we at least have `data`, `log` and `out` directories next time?

2. a earlier instruction: please take those who start early into account.

Thank you.
