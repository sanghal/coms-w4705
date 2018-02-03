COMS W4705 PSET #4
------------------

- Hang Gao (uni: 2469)
- Email: [hang.gao@columbia.edu](mailto:hang.gao@columbia.edu)
- Date: Dec 09, 2017


Mannual
-------

* Question 1
* Question 2
* Question 3


QUESTION 1
----------

1. Setup:

    > python src/q1.py

    > python src/depModel.py q1 model/q1.model trees/dev.conll \
        outputs/dev_part1.conll

    > python src/depModel.py q1 model/q1.model trees/test.conll \
        outputs/test_part1.conll

2. Result:

* model/q1.model: the learnt weights for defined model.

* outputs/dev_part1.conll: the parsed tree using q1-specified parameters on 
the dev set.

* outputs/test_part1.conll: the parsed tree using q1-specified parameters on 
the test set.

3. Performance

    > python src/eval.py trees/dev.conll outputs/dev_part1.conll

        Unlabeled attachment score 83.09
        Labeled attachment score 79.93

4. Observations

* Our baseline achieves ~.83 on unlabeled attachment and ~.80 on labeled
attachment which is good but this cannot demonstrate NN's model capacity is 
better than others (since log-linear could attach basically the same 
performance). Anyway, this is our baseline to improve through next enhancements.


QUESTION 2
----------

1. Setup:

    > python src/q2.py

    > python src/depModel.py q2 model/q2.model trees/dev.conll \
        outputs/dev_part2.conll

    > python src/depModel.py q2 model/q2.model trees/test.conll \
        outputs/test_part2.conll

2. Result:

* model/q2.model: the learnt weights for defined model.

* outputs/dev_part2.conll: the parsed tree using q2-specified parameters on 
the dev set.

* outputs/test_part2.conll: the parsed tree using q2-specified parameters on 
the test set.

3. Performance

    > python src/eval.py trees/dev.conll outputs/dev_part2.conll

        Unlabeled attachment score 84.24
        Labeled attachment score 80.99

4. Observations

* Through only adjusting the embedding dimensions of words' and POS labels' 
feature, we gain one percent of improvement on both unlabeled and labeled 
attachment.

* Yet the training duration is about 1.5-2 times of original experiment mainly
because that there's about twice the computations going on when updating the 
graph, and we are running the script on CPU machine therefore no parallelization 
could be executed.

* Back to the performance: in my opinion, this could be explained by the fact 
that there's now a higher dimensional space to compress the original one-hot 
information, thus pushing down the compression loss. With better embedding 
representations, it's natural to achieve better results.


QUESTION 3
----------

0. Explorations:

In this quesiton, I've explored mainly three ways to further improve NN model's 
performance:

    (1) Using SELU rather than RELU as activation for all layers. (Not working)
        SELU is proposed early this year on Arxiv. It was claimed to have self-
        normalizing property so that smoother update would be made between every 
        steps. I'm very curious about this simple technique and thus gave it a 
        shot as first attempt.

            Unlabeled attachment score 83.53
            Labeled attachment score 79.91

    (2) Adding dropout with probability of .5 before softmax. (Not working)
        Since our previous model exhibited certain degree of over-fitting (1 on
        training set and ~.84 on develop set), it is natural to add randomized 
        noise as a kind of regularization. Here I chose Dropout to do so, 
        however I'm not 100% sure about whether it's optimal for such a shallow 
        network.

            Unlabeled attachment score 82.23
            Labeled attachment score 79.17

    (3) Initialize word embedding layer with pre-trained word vectors. (Worked)
        I chose `glove.6B.100d` word vectors to initialize our embedding layer. 
        For the non-overlapped words, I just initialized their weights with 
        all zeros.

            Unlabeled attachment score 85.0
            Labeled attachment score 81.84

The final result is reported with only "pre-trained wordvec" setting. To 
make the results more comparable with previous experiments, I kept the rest of 
parameters the same.

1. Setup:

    > python src/q3.py

    > python src/depModel.py q3 model/q3.model trees/dev.conll \
        outputs/dev_part3.conll

    > python src/depModel.py q3 model/q3.model trees/test.conll \
        outputs/test_part3.conll

2. Result:

* model/q3.model: the learnt weights for defined model.

* outputs/dev_part3.conll: the parsed tree using q2-specified parameters on 
the dev set.

* outputs/test_part3.conll: the parsed tree using q2-specified parameters on 
the test set.

3. Performance

    > python src/eval.py trees/dev.conll outputs/dev_part3.conll

        Unlabeled attachment score 85.0
        Labeled attachment score 81.84

4. Observations

* Although we have tried lines of method, only using the pretrained word vector 
worked in this case, with around ~1% improvement over develop set for both 
unlabeled and labeled scenario.

* The reason why it works is simple: we transferred knowledge from wider dataset 
(Twitter 400k & Wikipedia 100k in this case) to our task. With the help of more 
data collect and the hidden relationship inside them, we can surly learn better 
model for POS.

* The experiments listed is only a portion of our exploration. We've even tried 
different parameters for different variables but all resulting in non-improvement 
which was quite frustrating. With more time given, we would like to build more 
complicated model like LSTM for this task which is recently proved to work very 
well rather than being exhausted tuning a simple 2-hidden-layer NN.
