# Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems

![](images/book.png)

Authors: Martin Kleppmann


## Introduction

The goal of this book is to help you navigate the diverse and fast-changing landscape of technologies for processing and storing data. This book is not a tutorial for one particular tool, nor is it a textbook full of dry theory. Instead, we will look at examples of successful data systems: technologies that form the foundation of many popular applications and that have to meet scalability, performance, and reliability requirements in production every day.

## PART I: Foundations of Data Systems

### CHAPTER 1: Reliable, Scalable, and Maintainable Applications
In this chapter, we will start by exploring the fundamentals of what we are trying to achieve: reliable, scalable, and maintainable data systems. We’ll clarify what those things mean, outline some ways of thinking about them, and go over the basics that we will need for later chapters.

#### Reliability
Reliability means that the system should continue to work correctly, even when things go wrong.

#### Scalability
Scalability is the term we use to describe a system’s ability to cope with increased load. Load can be described with a few numbers which we call load parameters. The best choice of parameters depends on the architecture of your system: it may be requests per second to a web server, the ratio of reads to writes in a database, the number of simultaneously active users in a chat room, the hit rate on a cache, or something else.

#### Maintainability
Maintainability means designing software in such a way that it will hopefully minimize pain during maintenance, and thus avoid creating legacy software. To this end, we will pay particular attention to three design principles for software systems:

**1- Operability:** Make it easy for operations teams to keep the system running smoothly.

**2- Simplicity:** Make it easy for new engineers to understand the system.

**3- Evolvability:** Make it easy for engineers to make changes to the system in the future.

#### Summary
In this chapter, we have explored some fundamental ways of thinking about data intensive applications. These principles will guide us through the rest of the book, where we dive into deep technical detail.

### CHAPTER 2: Data Models and Query Languages
Data models are perhaps the most important part of developing software, because they have such a profound effect: not only on how the software is written, but also on how we think about the problem that we are solving. In this chapter we will look at a range of general-purpose data models for data storage and querying. In particular, we will compare the relational model, the document model, and a few graph-based data models. We will also look at various query languages and compare their use cases.

#### Relational Model Versus Document Model

##### The Object-Relational Mismatch
The document model (e.g., JSON) often maps better to application code, especially for data with a tree-like structure. For example, an entire user profile with nested positions and education can be stored in a single document. This is simpler to retrieve than a relational model, which would require joining multiple tables (users, positions, education) to reconstruct the same profile.

##### Many-to-One and Many-to-Many Relationships
In relational databases, it’s normal to refer to rows in other tables by ID, because joins are easy. In document databases, joins are not needed for one-to-many tree structures, and support for joins is often weak.

##### Are Document Databases Repeating History?
When it comes to representing many-to-one and many-to-many relationships, relational and document databases are not fundamentally different: in both cases, the related item is referenced by a unique identifier, which is called a foreign key in the relational model and a document reference in the document model. That identifier is resolved at read time by using a join or follow-up queries.

##### Relational Versus Document Databases Today
The main arguments in favor of the document data model are schema flexibility, better performance due to data locality, and that for some applications it is closer to the data structures used by the application. The relational model counters by providing better support for joins, and many-to-one and many-to-many relationships.

It’s not possible to say in general which data model leads to simpler application code; it depends on the kinds of relationships that exist between data items. For highly interconnected data, the document model is awkward, the relational model is acceptable, and graph models are the most natural.

It seems that relational and document databases are becoming more similar over time, and that is a good thing: the data models complement each other. If a database is able to handle document-like data and also perform relational queries on it, applications can use the combination of features that best fits their needs.

#### Query Languages for Data
This section explores the fundamental difference between two types of query languages: **imperative and declarative**.

In **declarative** query languages (like SQL), you specify *what* data you want, but not *how* to retrieve it. The database's query optimizer decides the most efficient way to execute the query, which often leads to simpler code and better performance.

In contrast, **imperative** languages (like MapReduce) require you to provide step-by-step instructions on how to process the data. This gives more control but can be more complex and prevents the database from optimizing the execution plan.

This section highlights that this declarative approach is so powerful that it has been adopted for other data models, such as graph databases.

#### Graph-Like Data Models
While relational and document models work well for many data structures, they become cumbersome and inefficient when dealing with highly interconnected data, especially complex many-to-many relationships.

Graph data models are designed to address this by treating relationships as first-class citizens. A graph consists of two key components:
*   **Nodes** (or vertices): Represent entities (e.g., people, products).
*   **Edges** (or relationships): Represent the connections between nodes.

Crucially, both nodes and edges can have properties, allowing you to store rich information not just about the entities but also about the relationships themselves (e.g., the timestamp of a 'LIKES' relationship). This model makes traversing complex networks (like finding "friends of friends") intuitive and efficient.

The chapter introduces two main graph models: the **property graph model** (queried with languages like **Cypher**) and the **triple-store model** (queried with **SPARQL**). Both use a declarative approach, where you describe the pattern of nodes and relationships you want to find. The chapter also mentions **Datalog** as a more powerful, foundational data model that has influenced these modern query languages.

Graph models are the natural choice for use cases where relationships are central, such as social networks, recommendation engines, fraud detection, and knowledge graphs.

#### Summary
Data models are a powerful tool for dealing with the complexity of software, and different models are suited to different applications. The relational model has been dominant for a long time, but document and graph databases are now gaining popularity for their ability to handle specific use cases more naturally. The choice of a data model depends heavily on the structure of the data and the relationships within it. Furthermore, the move toward declarative query languages like SQL, Cypher, and SPARQL has made it easier to work with these models by allowing developers to focus on *what* they want, not *how* to get it.

### CHAPTER 3


### CHAPTER 4


## PART II: Distributed Data



## PART III: Derived Data

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
Data models are perhaps the most important part of developing software, because
they have such a profound effect: not only on how the software is written, but also on
how we think about the problem that we are solving.
In this chapter we will look at a range of general-purpose data models for data stor‐
age and querying.  In particular, we will compare the relational model, the document model, and a few graph-based data models. We will
also look at various query languages and compare their use cases.


#### Relational Model Versus Document Model


##### The Object-Relational Mismatch

The JSON representation has better locality than the multi-table schema in Figure below. If you want to fetch a profile in the relational example, you need to either
perform multiple queries (query each table by user_id) or perform a messy multi way join between the users table and its subordinate tables. In the JSON representation, all the relevant information is in one place, and one query is sufficient.

```json
{
  "user_id": 251,
  "first_name": "Bill",
  "last_name": "Gates",
  "summary": "Co-chair of the Bill & Melinda Gates... Active blogger.",
  "region_id": "us:91",
  "industry_id": 131,
  "photo_url": "/p/7/000/253/05b/308dd6e.jpg",
  "positions": [
    {
      "job_title": "Co-chair",
      "organization": "Bill & Melinda Gates Foundation"
    },
    {
      "job_title": "Co-founder, Chairman",
      "organization": "Microsoft"
    }
  ],
  "education": [
    {
      "school_name": "Harvard University",
      "start": 1973,
      "end": 1975
    },
    {
      "school_name": "Lakeside School, Seattle",
      "start": null,
      "end": null
    }
  ],
  "contact_info": {
    "blog": "http://thegatesnotes.com",
    "twitter": "http://twitter.com/BillGates"
  }
}
```


##### Many-to-One and Many-to-Many Relationships
In relational databases, it’s normal to refer
to rows in other tables by ID, because joins are easy. In document databases, joins are
not needed for one-to-many tree structures, and support for joins is often weak.


##### Are Document Databases Repeating History?

While many-to-many relationships and joins are routinely used in relational databases, document databases and NoSQL reopened the debate on how best to represent
such relationships in a database. 

When it comes to representing many-to-one and many-to-many relationships, relational and document databases are not fundamentally different: in both
cases, the related item is referenced by a unique identifier, which is called a foreign
key in the relational model and a document reference in the document model.
That identifier is resolved at read time by using a join or follow-up queries.


##### Relational Versus Document Databases Today

The main arguments in favor of the document data model are schema flexibility, better performance due to locality, and that for some applications it is closer to the data
structures used by the application. The relational model counters by providing better
support for joins, and many-to-one and many-to-many relationships.
The poor support for joins in document databases may or may not be a problem,
depending on the application. For example, many-to-many relationships may never
be needed in an analytics application that uses a document database to record which
events occurred at which time. However, if your application does use many-to-many relationships, the document model becomes less appealing.

It’s not possible to say in general which data model leads to simpler application code;
it depends on the kinds of relationships that exist between data items. For highly
interconnected data, the document model is awkward, the relational model is acceptable, and graph models are the most natural.

It seems that relational and document databases are becoming more similar over
time, and that is a good thing: the data models complement each other. If a database
is able to handle document-like data and also perform relational queries on it, applications can use the combination of features that best fits their needs.

#### Query Languages for Data

This section explores the fundamental difference between two types of query languages: **imperative and declarative**.

In declarative query languages (like SQL), you specify what data you want, but not how to retrieve it. The database's query optimizer decides the most efficient way to execute the query, which often leads to simpler code and better performance.

In contrast, imperative languages (like MapReduce) require you to provide step-by-step instructions on how to process the data. This gives more control but can be more complex and prevents the database from optimizing the execution plan.

This section highlights that this declarative approach is so powerful that it has been adopted for other data models, such as graph databases. 




#### Graph-Like Data Models

While relational and document models work well for many data structures, they become cumbersome and inefficient when dealing with highly interconnected data, especially complex many-to-many relationships.

Graph data models are designed to address this by treating relationships. A graph consists of two key components:

- Nodes (or vertices): Represent entities (e.g., people, products).
- Edges (or relationships): Represent the connections between nodes.

Crucially, both nodes and edges can have properties, allowing you to store rich information not just about the entities but also about the relationships themselves (e.g., the timestamp of a 'LIKES' relationship). This model makes traversing complex networks (like finding "friends of friends") intuitive and efficient.

This section introduces two main graph models: the property graph model (queried with languages like Cypher) and the triple-store model (queried with SPARQL). Both use a declarative approach, where you describe the pattern of nodes and relationships you want to find.

Graph models are the natural choice for use cases where relationships are central, such as social networks, recommendation engines, fraud detection, and knowledge graphs.





### CHAPTER 3


### CHAPTER 4


## PART II: Distributed Data



## PART III: Derived Data