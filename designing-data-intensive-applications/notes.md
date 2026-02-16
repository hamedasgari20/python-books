# Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems

![](images/book.png)

Authors: Martin Kleppmann


## Introduction

The goal of this book is to help you navigate the diverse and fast-changing landscape
of technologies for processing and storing data. This book is not a tutorial for one
particular tool, nor is it a textbook full of dry theory. Instead, we will look at examples
of successful data systems: technologies that form the foundation of many popular
applications and that have to meet scalability, performance, and reliability requirements in production every day.

## PART I: Foundations of Data Systems

### CHAPTER 1: Reliable, Scalable, and Maintainable Applications
In this chapter, we will start by exploring the fundamentals of what we are trying to
achieve: reliable, scalable, and maintainable data systems. We’ll clarify what those
things mean, outline some ways of thinking about them, and go over the basics that
we will need for later chapters.

#### Reliability
Reliability means that the system should continue to work correctly, even when things go wrong.

#### Scalability
Scalability is the term we use to describe a system’s ability to cope with increased load. Load can be described
with a few numbers which we call load parameters. The best choice of parameters
depends on the architecture of your system: it may be requests per second to a web
server, the ratio of reads to writes in a database, the number of simultaneously active
users in a chat room, the hit rate on a cache, or something else.


#### Maintainability
Maintainability means designing software in such a way that it will hopefully minimize pain during maintenance, and thus avoid creating legacy software
To this end, we will pay particular attention to three design principles for software systems:

**1- Operability:** Make it easy for operations teams to keep the system running smoothly.

**2- Simplicity:** Make it easy for new engineers to understand the system.

**3- Evolvability:** Make it easy for engineers to make changes to the system in the future.

#### Summary
In this chapter, we have explored some fundamental ways of thinking about data intensive applications. These principles will guide us through the rest of the book,
where we dive into deep technical detail.


### CHAPTER 2


### CHAPTER 3


### CHAPTER 4


## PART II: Distributed Data



## PART III: Derived Data