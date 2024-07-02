# Prompting

Prompting is a challenging task, with many small nuances that we need to take note of. We've compiled a list of 58 different prompting techniques that you can take advantage of today in order to get a quick boost to your model's performance.

We've separated them into the following categories - [**Zero Shot**](#zero-shot), [**Few Shot**](#few-shot), [**Thought Generation**](#thought-generation), [**Ensembling**](#ensembling), [**Self-Criticism**](#self-criticism) and [**Decomposition**](#decomposition)

Each of these techniques offers unique advantages in different scenarios. Click on the links to learn more about each method and how to apply them effectively in your prompts.

## Zero-Shot

Before you get any examples, how can you maximise the effectiveness of your prompt? Zero Shot techniques help us to do so well.

1. [Emotion Prompting](zero_shot/emotion_prompting.md)
2. [Role Prompting](zero_shot/role_prompting.md)
3. [Style Prompting](zero_shot/style_prompting.md)
4. [S2A (Sentence to Action)](zero_shot/s2a.md)
5. [SimToM (Simulated Theory of Mind)](zero_shot/simtom.md)
6. [RaR (Retrieval-augmented Response)](zero_shot/rar.md)
7. [RE2 (Recursive Explanation and Elaboration)](zero_shot/re2.md)
8. [Self-Ask](zero_shot/self-ask.md)

## Few-Shot

When choosing examples, how can we ensure they make a big difference in our model's performance? This isn't an easy thing to do and so we've broken it down into a few different things

1. [SG-ICL](few_shot/example_generation/sg_icl.md)
2. [Example Ordering](few_shot/example_ordering.md)
3. [KNN Choice](few_shot/exemplar_selection/knn.md)
4. [Vote-K](few_shot/exemplar_selection/vote_k.md)

## Thought Generation

How can we encourage our model to reason better to get to the final result?

1. [Analogical Prompting](thought_generation/chain_of_thought_zero_shot/analogical_prompting.md)
2. [Step-Back Prompting](thought_generation/chain_of_thought_zero_shot/step_back_prompting.md)
3. [Thread-of-Thought (ThoT)](thought_generation/chain_of_thought_zero_shot/thread_of_thought.md)
4. [Tab-CoT](thought_generation/chain_of_thought_zero_shot/tab_cot.md)
5. [Active-Prompt](thought_generation/chain_of_thought_few_shot/active_prompt.md)
6. [Auto-CoT](thought_generation/chain_of_thought_few_shot/auto_cot.md)
7. [Complexity-Based](thought_generation/chain_of_thought_few_shot/complexity_based.md)
8. [Contrastive](thought_generation/chain_of_thought_few_shot/contrastive.md)
9. [Memory-of-Thought](thought_generation/chain_of_thought_few_shot/memory_of_thought.md)
10. [Uncertainty-Routed CoT](thought_generation/chain_of_thought_few_shot/uncertainty_routed_cot.md)
11. [Prompt Mining](thought_generation/chain_of_thought_few_shot/prompt_mining.md)

## Ensembling

How can we combine multiple parallel inference calls to get a significant boost in performance? Ensembling techniques allow us to leverage the strengths of multiple model runs, potentially leading to more accurate and robust results.

1. [COSP](ensembling/cosp.md)
2. [DENSE](ensembling/dense.md)
3. [DiVeRSe](ensembling/diverse.md)
4. [Max Mutual Information](ensembling/max_mutual_information.md)
5. [Meta-CoT](ensembling/meta_cot.md)
6. [MoRE](ensembling/more.md)
7. [Self-Consistency](ensembling/self_consistency.md)
8. [Universal Self-Consistency](ensembling/universal_self_consistency.md)
9. [USP](ensembling/usp.md)
10. [Prompt Paraphrasing](ensembling/prompt_paraphrasing.md)

## Self-Criticism

What concrete steps can we take to get our model to critically evaluate and improve its own outputs? Self-criticism methods encourage the model to evaluate and refine its responses, promoting higher quality and more thoughtful outputs.

1. [Chain-Of-Verification](self_criticism/chain_of_verification.md)
2. [Self-Calibration](self_criticism/self_calibration.md)
3. [Self-Refine](self_criticism/self_refine.md)
4. [Self-Verification](self_criticism/self_verification.md)
5. [ReverseCoT](self_criticism/reversecot.md)
6. [Cumulative Reason](self_criticism/cumulative_reason.md)

## Decomposition

How can we break down complex problems into more manageable parts? Decomposition prompting methods offer an effective strategy to approach intricate questions by dividing them into smaller, more manageable sub-questions, allowing for a more structured and comprehensive problem-solving approach.

1. [DECOMP](decomposition/decomp.md)
2. [Faithful CoT](decomposition/faithful_cot.md)
3. [Least-to-Most](decomposition/least_to_most.md)
4. [Plan-and-Solve](decomposition/plan_and_solve.md)
5. [Program-of-Thought](decomposition/program_of_thought.md)
6. [Recurs.-of-Thought](decomposition/recurs_of_thought.md)
7. [Skeleton-of-Thought](decomposition/skeleton_of_thought.md)
8. [Tree-of-Thought](decomposition/tree-of-thought.md)

# <sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
