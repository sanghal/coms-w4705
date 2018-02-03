COMS W4705 PSET #2
------------------

- Hang Gao (UNI: 2469)
- Email: [hang.gao@columbia.edu](mailto:hang.gao@columbia.edu)
- Date: Oct 28, 2017


Mannual
-------

* Question 4
* Question 5
* Question 6


QUESTION 4
----------

1. Setup:

    > python parser.py q4 parse_train.dat parse_train.RARE.dat

    **** With total running time of 4.01s

2. Result:

    * parse_train.RARE.dat: the training data file with new counts, by replacing 
    infrequent words with `_RARE_`.


QUESTION 5
----------

1. Setup:

    > python parser.py q5 parse_train.RARE.dat parse_dev.dat q5_prediction_file

    **** With total running time of 44.29s

    > python eval_parser.py parse_dev.key q5_prediction_file > q5_eval.txt

2. Result:

    * q5_prediction_file: the parsing result for `parse_dev.dat`

    * q5_eval.txt: eval result for question 5

3. Performance

    > cat q5_eval.txt

          Type       Total   Precision      Recall     F1 Score
    ===============================================================
             .         370     1.000        1.000        1.000
           ADJ         164     0.827        0.555        0.664
          ADJP          29     0.333        0.241        0.280
      ADJP+ADJ          22     0.542        0.591        0.565
           ADP         204     0.955        0.946        0.951
           ADV          64     0.694        0.531        0.602
          ADVP          30     0.333        0.133        0.190
      ADVP+ADV          53     0.756        0.642        0.694
          CONJ          53     1.000        1.000        1.000
           DET         167     0.988        0.976        0.982
          NOUN         671     0.752        0.842        0.795
            NP         884     0.625        0.524        0.570
        NP+ADJ           2     0.286        1.000        0.444
        NP+DET          21     0.783        0.857        0.818
       NP+NOUN         131     0.641        0.573        0.605
        NP+NUM          13     0.214        0.231        0.222
       NP+PRON          50     0.980        0.980        0.980
         NP+QP          11     0.667        0.182        0.286
           NUM          93     0.984        0.645        0.779
            PP         208     0.593        0.630        0.611
          PRON          14     1.000        0.929        0.963
           PRT          45     0.957        0.978        0.967
       PRT+PRT           2     0.400        1.000        0.571
            QP          26     0.647        0.423        0.512
             S         587     0.626        0.782        0.695
          SBAR          25     0.091        0.040        0.056
          VERB         283     0.683        0.799        0.736
            VP         399     0.559        0.594        0.576
       VP+VERB          15     0.250        0.267        0.258

         total        4664     0.713        0.713        0.713

4. Observations

    * Our baseline model has the total precision, recall and according 
    F1-score at .713.

    * Noticed that there exists a strong deviation between different 
    non-terminals: it parsed extremely well for ``naive'' ones - `.`, `CONJ`, 
    `PRON`, `DET` and so on - which could be explained by the fact that 
    ambiguity doesn't occur much for these categories; on the other hand, 
    extremely poor counterpart also exists - `VP+VERB`, `ADJP`, `NP+ADJ`, 
    `NP+NUM` and etc. Of course, for `ADJP`, it could be interpreted as the 
    turbulence of ambiguity, what's more interesting is that the ``combined'' 
    non-terminals are prone to yield worse results. One possible problem might 
    be with the CKY formalization.

    * Also note that the performance shown that our baseline parser is a 
    balanced model since the scale of the most metrics are the same.


QUESTION 6
----------

1. Setup:

    > python parser.py q4 parse_train_vert.dat parse_train_vert.RARE.dat

    **** With total running time of 4.12s

    > python parser.py q6 parse_train_vert.RARE.dat parse_dev.dat q6_prediction_file

    **** With total running time of 94.89s

    > python eval_parser.py parse_dev.key q6_prediction_file > q6_eval.txt

2. Result:

    * parse_train_vert.RARE.dat: the training data file with new counts, by 
    replacing infrequent words with `_RARE_`.

    * q6_prediction_file: the parsing result for `parse_dev.dat`.

    * q6_eval.txt: eval result for question 6.

3. Performance

    > cat q5_eval.txt

          Type       Total   Precision      Recall     F1 Score
    ===============================================================
             .         370     1.000        1.000        1.000
           ADJ         164     0.689        0.622        0.654
          ADJP          29     0.324        0.414        0.364
      ADJP+ADJ          22     0.591        0.591        0.591
           ADP         204     0.960        0.951        0.956
           ADV          64     0.759        0.641        0.695
          ADVP          30     0.417        0.167        0.238
      ADVP+ADV          53     0.700        0.660        0.680
          CONJ          53     1.000        1.000        1.000
           DET         167     0.988        0.994        0.991
          NOUN         671     0.795        0.845        0.819
            NP         884     0.617        0.548        0.580
        NP+ADJ           2     0.333        0.500        0.400
        NP+DET          21     0.944        0.810        0.872
       NP+NOUN         131     0.610        0.656        0.632
        NP+NUM          13     0.375        0.231        0.286
       NP+PRON          50     0.980        0.980        0.980
         NP+QP          11     0.750        0.273        0.400
           NUM          93     0.914        0.688        0.785
            PP         208     0.623        0.635        0.629
          PRON          14     1.000        0.929        0.963
           PRT          45     1.000        0.933        0.966
       PRT+PRT           2     0.286        1.000        0.444
            QP          26     0.650        0.500        0.565
             S         587     0.704        0.814        0.755
          SBAR          25     0.667        0.400        0.500
          VERB         283     0.790        0.813        0.801
            VP         399     0.663        0.677        0.670
       VP+VERB          15     0.294        0.333        0.312

         total        4664     0.742        0.742        0.742

4. Observations

    * At this section, we gained a solid improvement at ~.3 than the baseline, 
    We could see that the prediction of some types leaps a lot(e.g. `SBAR`, 
    precision and recall improve from < .1 to > .4).
    
    * Some of previously weak ones like `VP+VERB`, `ADJP+ADJ` only gain a bit 
    improvement around 2 percent.

    * The improvement could be largely accounted that we now incorporate 
    higher dependency and contextual information, so that the strong assumption 
    could be relaxed. The vertical Markovation helped by encoding the 
    information of parent non-terminals, so that it has greater capability 
    distinguishing the non-terminals and increase the flexibility of rules. 
    Since the number of rules also increased during procession, a larger 
    training corpus would be needed.

