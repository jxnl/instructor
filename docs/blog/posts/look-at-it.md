---
draft: False
date: 2023-11-26
slug: python-caching
tags:
  - caching
  - functools
  - diskcache
  - python
  - rag
authors:
  - jxnl
---

# Improving your RAG with Question Types

!!! note "This is a work in progress"

    This is a work in progress I'm going to use bullet points to outline the main points of the article.

It is my intention that by the end of this blog post you will have some tools to help you improve your RAG that are not just "try a different chunk size" or "try a different re-ranker". These are all great things to try, but I want to give you some tools to help you understand your data and make informed decisions on how to improve your RAG.

But before I do that, let's go over the generic advice so we can get that out of the way.

## Evals

The point of having evaluations is not to have an absolute metric to determine your success. The best way to improve is to have a set of evals that execute quickly, which allows you to spin up and prepare many experiments. Once we run them, the goal is that we are only looking for changes in a metric. It does not matter if the metric is a 3 or a 4. What matters is that the 3 can improve to a 3.5 and the 4 might improve to a 4.5.

Evaluations help you iterate. And the faster you can evaluate, the more experiments you can run on your coffee break.

## Rerankers

Every company I've worked with so far has tried something like Kohiroi Rancors and have generally liked it very much. I don't have much to say here except for the fact that if you haven't tried it already, definitely consider running these and comparing how your metrics improve.

## Parameter Sweeps

This is a very generic piece of advice, run parameter sweeps. It's very common in machine learning to train dozens of models and just simply pick the best one for some metric. This is not any different in building these retrieval applications.

If you want to spend an hour writing a design doc and how you might want to test different chunk sizes, you may as well just prepare the test suite. And being able to chunk differently and chunk quickly and define different overlaps and re-rankers will allow you to iterate by just writing these giant jobs over the weekends. Your only constraint really is going to be money and time.

This could mean trying 200, 400, 800 chunk size with 20% and 50% overlap, Putting that in a queue and running it over the weekend. The trick here really is doing things quickly. I prefer to use tools like [Modal Labs](modal.com) to quickly spin up these experiments and run them in parallel. Then my only constraint is money. And if you're building something that makes money, this shouldn't be a problem.

## Summary Index

In a production-ranked application, we can't guarantee that the LLM will be able to answer the question correctly every time. But what we can do is give confidence to the user that the documents that are relevant are being shown. An easy way of doing that is to run some kind of summarization like [chain of density](./chain-of-density.md), embedding that summary, and doing retrieval of documents first, and then putting them into context rather than using chunks themselves.

## Specific Advice

This part becomes much more challenging because these issues are not going to be solved by simply playing around with re-rankers and chunk size. It requires examining your data and applying a range of data science techniques to gain a deep understanding of the data. This will enable us to make informed decisions on how to construct new indices and develop specific search engines that cater to your customers' needs.

Once we analyze the data, we will have ample information to identify the most effective interventions and areas where specialization is necessary.

In this blog we'll cover a range of things that can lead us into the right direction. Go over some examples of companies that can do this kind of exploration. We'll leave it open ended as to what the interventions are, but give you the tools to drill down into your data and figure out what you need to do. For example if if google learns that a large portion of queries are looking for directions or a location, they might want to build a seperate index and release a maps product rather than expecting a HTML page to be the best response.

## What do I look for?

- The first time we shuold be looking is simply looking at the questions we're asking.
- Find some inductive bias in the questions we're asking.
- IF we have a general idea (we could even do topic model, but we can cover that later) we can start to look at the questions we're asking.
- Then we can build a question type classifier to help us identify the question type.
- We can look at two types of statics, the distribution of question types and the quality of the responses
- Then by looking at these two quantities we can determine our intervention strategy.

1. IF counts is high and quality is low, we likle have to do something to improve the quality of the responses.
2. if counts is low and quality is low, we might just want to add in some logic that says "if question type is X, then do don't answer it and give some canned response"

### Consider Google.

You can imagine day one of google, they can spend tonnes of time looking at the data and trying to figure out what to do, but they can also just look at the data and see what people are searching for.

we might discover that there are tonnes of search questions that look like directions that doo poorly because there are only few websites that give directinos from one place to another, so they identiy that they might want to support a maps feature, same with photos, or shopping or videos.

we might also notice that for sports games and showtimes, and weather, they might want to return smaller modals rather than a complete new 'page' of results. All these decisions are likely something that could be done by inspecting the data early on in the business

## Unreasonable Effectiveness of Looking

Once you've looked you'll usually break it down into two categories:

1. Topics
2. Capabilities

### Topics

I see topics as the data that could be retrieved via text or semantic search. For embedding search it could be the types of text that isearched. "Privacy documents, legal documents, etc" are all topics since they can completed be generated by a searc hquery.

**Failure Modes**

1. Poor inventory: Usually the topics fail its a result of poor inventory. If 50% of your queries are about privacy documents and you dont have any privacy documents in your inventory, then you're going to have a bad time.
2. Query Mismatch: This could be as simple as queries for "GDPR pocily" and "Data Privacy Policy" are both about the same topic, but based on the search method you might not be able to find the right documents.

### Capabilities

Capabilities are the things that you can do with your index itself. For example if you have a plain jane search index only over text. being able to answer comparisone questions and timeline questions are going to be capabilities that you can't do unless you bake them into your index. otherwise we'll embed to somethign strange "What happened last week" NEEDS to be embedded to a date, otherwise you're going to have a bad time. This is somethign we covered a lot [Rag is more than embedding search](./rag-and-beyond.md).

Heres some more examples of capabilities:

1. Ownership of content: "Who uploaded the XYZ document about ABC"
2. Timeline queries: "What happened last week"
3. Comparisons: "What is the difference between X and Y"
4. Directions: "How do I get from X to Y"
5. Content Type: "Show me videos about X"
6. Document Metadata: "Show me documents that were created by X in the last week"

These are all families of queries that cannot be solved via the embedding and require creative solutions based on your use case!

**Failure Modes**

- You literally just can't service many of the unless you bake them into your index.
- If you don't have a maps index, you can't answer most directions questions.
- If you don't have a video -> transcript index, you can't answer most video questions.
- If you don't have a time based query capability, you can't answer timeline questions.

## Eval Benefits of classifying by capability

Once you've classified your questions by capability, you can also start specializing not only to generate response using specific capabilities, but also specialize your evaluation metrics. Imagine you're a construction site, and you have 'ownership' questions. If you knew that apriori, you'd know to add in the prompt that the responses of 'ownership' must result in not only the name of a person, but a full name and contact information. You can then, also validate in an eval if that criteria is met!
