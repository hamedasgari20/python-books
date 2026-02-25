# Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems

![](images/book.png)

Authors: Martin Kleppmann

<!-- TOC -->
* [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](#designing-data-intensive-applications-the-big-ideas-behind-reliable-scalable-and-maintainable-systems)
  * [Introduction](#introduction)
  * [PART I: Foundations of Data Systems](#part-i-foundations-of-data-systems)
    * [CHAPTER 1: Reliable, Scalable, and Maintainable Applications](#chapter-1-reliable-scalable-and-maintainable-applications)
      * [Reliability](#reliability)
      * [Scalability](#scalability)
      * [Maintainability](#maintainability)
      * [Summary](#summary)
    * [CHAPTER 2: Data Models and Query Languages](#chapter-2-data-models-and-query-languages)
      * [Relational Model Versus Document Model](#relational-model-versus-document-model)
        * [The Object-Relational Mismatch](#the-object-relational-mismatch)
        * [Many-to-One and Many-to-Many Relationships](#many-to-one-and-many-to-many-relationships)
        * [Are Document Databases Repeating History?](#are-document-databases-repeating-history)
        * [Relational Versus Document Databases Today](#relational-versus-document-databases-today)
      * [Query Languages for Data](#query-languages-for-data)
      * [Graph-Like Data Models](#graph-like-data-models)
      * [Summary](#summary-1)
    * [CHAPTER 3: Storage and Retrieval](#chapter-3-storage-and-retrieval)
      * [Data Structures That Power Your Database](#data-structures-that-power-your-database)
        * [Hash Indexes](#hash-indexes)
        * [SSTables and LSM-Trees](#sstables-and-lsm-trees)
        * [B-Trees](#b-trees)
        * [Comparing B-Trees and LSM-Trees](#comparing-b-trees-and-lsm-trees)
        * [Other Indexing Structures](#other-indexing-structures)
      * [Transaction Processing or Analytics?](#transaction-processing-or-analytics)
        * [Data Warehousing](#data-warehousing)
        * [Stars and Snowflakes: Schemas for Analytics](#stars-and-snowflakes-schemas-for-analytics)
      * [Column-Oriented Storage](#column-oriented-storage)
        * [Column Compression](#column-compression)
        * [Sort Order in Column Storage](#sort-order-in-column-storage)
        * [Writing to Column-Oriented Storage](#writing-to-column-oriented-storage)
        * [Aggregation: Data Cubes and Materialized Views](#aggregation-data-cubes-and-materialized-views)
      * [Summary](#summary-2)
    * [CHAPTER 4](#chapter-4)
  * [PART II: Distributed Data](#part-ii-distributed-data)
  * [PART III: Derived Data](#part-iii-derived-data)
<!-- TOC -->


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
Data models are fundamental to how we think about problems and write software. This chapter compares the most common models—relational, document, and graph—and the query languages used to interact with them.

#### Relational Model Versus Document Model

The core difference lies in how data is structured. The **relational model** normalizes data into multiple tables, linked by IDs (foreign keys). The **document model** stores related data in a single, tree-like document (e.g., JSON).

*   **Document Model Strengths:** Offers schema flexibility and better performance for "one-to-many" relationships, as all related data is read in one go (data locality).
*   **Relational Model Strengths:** Excels at handling "many-to-many" relationships and joins, which are often weak or cumbersome in document databases.

**Conclusion:** The choice depends on your data. For tree-like structures, documents are simpler. For highly interconnected data, relational is better, and graph models are best. Modern databases are increasingly adopting features from both models.

#### Query Languages for Data

This section covers the critical difference between **imperative** and **declarative** query languages.

*   **Imperative** (e.g., MapReduce): You specify *how* to get the data with step-by-step instructions.
*   **Declarative** (e.g., SQL): You specify *what* data you want, and the database's query optimizer figures out the most efficient way to retrieve it.

The declarative approach is superior because it leads to simpler code and allows the database to automatically optimize performance. This "what, not how" principle is now the foundation for most modern query languages, including those for graph databases.

#### Graph-Like Data Models

For highly interconnected data where relationships are as important as the entities themselves, **graph models** are the most natural fit.

A graph consists of:
*   **Nodes:** Entities (e.g., people, products).
*   **Edges:** Relationships between nodes (e.g., 'FRIENDS_WITH', 'BOUGHT').

Crucially, both nodes and edges can have properties (e.g., a 'FRIENDS_WITH' edge could have a `since` date). This model makes traversing complex networks (e.g., "find friends of friends") intuitive and efficient. Graph databases use declarative query languages like **Cypher** and **SPARQL** to describe patterns in the graph.

#### Summary

There is no single "best" data model. The key is to choose the right tool for the job:
*   Use **relational** databases for structured data and complex joins.
*   Use **document** stores for flexible, tree-like data.
*   Use **graph** databases for highly interconnected data.

The trend toward **declarative query languages** (like SQL, Cypher) allows developers to focus on the logic of their questions, leaving the performance optimization to the database engine.

### CHAPTER 3: Storage and Retrieval
Now that we've explored the logical structure of data in Chapter 2, Chapter 3 dives into the physical layer: how databases actually store and retrieve data on hardware. The way a database organizes data on disk has a profound impact on its performance for different types of tasks.
The chapter introduces a fundamental distinction between two types of workloads: Online Transaction Processing (OLTP) and Online Analytical Processing (OLAP)

We will explore the internal data structures that power these engines, such as B-Trees and LSM-Trees for transactional systems, and discover why column-oriented storage is the key to high-performance analytics. Ultimately, this chapter reveals the engineering trade-offs that determine why some databases are fast for writes, others for reads, and why different tools are needed for different jobs.

#### Data Structures That Power Your Database


##### Hash Indexes
Hash indexes are one of the simplest indexing structures, designed for fast key-value lookups.

They work by using a hash function to map a key to a specific position (or slot) in an in-memory array. This slot contains a pointer to the actual location of the data on disk.

The primary advantage is speed: to find a value, you compute the hash, go directly to the slot, and jump to the data on disk, avoiding a full file scan.

However, hash indexes have a major limitation: they are not suitable for range queries. Since the hash function scatters keys randomly, you cannot efficiently retrieve all keys within a specific range (e.g., all users with IDs between 100 and 200).

This issue motivates the need for more sophisticated structures like Log-Structured Merge-Trees (LSM-Trees), which are discussed next.



##### SSTables and LSM-Trees
This section introduces a more advanced storage structure designed to overcome the limitations of simple log files, such as file fragmentation and the inability to perform range queries.

**SSTable (Sorted String Table)**

An SSTable is a disk file where key-value pairs are sorted by key. Crucially, once an SSTable is written to disk, it is immutable (it never changes).

- **Advantage:** Because the data is sorted, it enables efficient range queries (e.g., finding all keys between 'a' and 'c').


**LSM-Tree (Log-Structured Merge-Tree)**

The LSM-Tree is the architecture that manages a collection of SSTables. It optimizes for write-heavy workloads.

**Write Path:**
When a write comes in, it is first added to an in-memory, sorted table called a MemTable. This is a very fast operation.
When the MemTable gets full, it is written to disk as a new, immutable SSTable.

**Read Path:**
To read a key, the system first checks the MemTable.
If the key isn't there, it searches the SSTables on disk, starting with the newest one.

**Compaction:**
A background process periodically merges and compacts SSTables. This process discards outdated or deleted keys, consolidates files, and keeps read performance from degrading over time.
The LSM-Tree structure makes writes extremely fast by batching them in memory before writing to disk, making it a popular choice for high-throughput systems like Cassandra and RocksDB.









##### B-Trees
In contrast to the LSM-Tree approach, B-Trees are a storage structure that keeps data sorted on disk at all times.

A B-Tree is essentially a self-balancing tree structure where each node (called a page) contains multiple keys and pointers to child nodes. These pages are designed to be the size of a disk block, making disk I/O very efficient. The tree is kept wide and shallow, meaning that finding any key requires only a few disk reads.

**Operations:**

- **Read:** To find a key, the system traverses from the root to a leaf, which is very fast and predictable.
- **Write:** When a key is updated or inserted, the system finds the appropriate page and modifies it. If a page becomes full, it splits into two, and the tree re-balances itself.

**Comparison to LSM-Trees:**

The key trade-off between B-Trees and LSM-Trees is in read vs. write 

**performance:**
B-Trees offer faster, more predictable reads because each key has a single location. However, writes can be slower due to the overhead of finding the right page and potential page splits.
LSM-Trees offer faster writes (by appending to a log in memory) but reads can be slower, as the system may need to check multiple locations (MemTable and several SSTables).
Because of their balanced performance for both reads and writes, B-Trees are the most common indexing structure in traditional relational databases like PostgreSQL, MySQL, and Oracle.



##### Comparing B-Trees and LSM-Trees

The choice between B-Trees and LSM-Trees involves a fundamental trade-off between read and write performance.

| Feature | B-Tree | LSM-Tree |
| :--- | :--- | :--- |
| **Write Speed** | Slower. Requires finding the correct page and potential page splits. | **Very Fast.** Simply appends to an in-memory log (MemTable). |
| **Read Speed** | **Faster & Predictable.** Each key has a single, known location. | Slower. Must check the MemTable and potentially multiple SSTables. |
| **Disk Overhead** | Lower. Only the modified page is rewritten. | Higher. Background compaction causes **write amplification** (rewriting data multiple times). |


**When to Use Which:**

- Use B-Trees for read-heavy workloads or when you need predictable performance for both reads and writes. This makes them the standard for most relational databases (e.g., PostgreSQL, MySQL).
- Use LSM-Trees for write-heavy workloads with very high write throughput, such as logging, time-series data, or IoT data ingestion. This is why they are the foundation of many NoSQL databases (e.g., Cassandra, RocksDB).



##### Other Indexing Structures

Beyond B-Trees and LSM-Trees, many specialized indexing structures exist to optimize for specific types of queries. The key takeaway is that no single index is best for all jobs.

**1. Clustered vs. Non-Clustered Indexes**
This distinction is about where the actual data rows are stored.

**2. Multi-dimensional Indexes (e.g., R-Tree)**
Standard indexes are ineffective for queries like "find all cafes within 500 meters." R-Trees solve this.
*   **How it works:** It groups nearby points on a map into rectangles, then groups those rectangles into larger rectangles, forming a tree structure.
*   **Use Case:** Geospatial data, allowing for efficient queries about location and proximity.

**3. Full-Text Search Indexes (e.g., Inverted Index)**
Finding a word inside a large block of text is a common challenge.
*   **How it works:** An inverted index is a map that takes a word and returns a list of all documents (or document IDs) containing that word. It works like the index at the back of a book.
*   **Use Case:** The foundation of search engines like Elasticsearch and Lucene, enabling fast keyword searches.

In summary, choosing the right tool for the job—like a B-Tree for values, an R-Tree for location, or an inverted index for text—is crucial for building high-performance applications.




#### Transaction Processing or Analytics?


This section introduces a fundamental distinction in how we use data: for operational transactions or for historical analysis. These two patterns, called OLTP (Online Transaction Processing) and OLAP (Online Analytical Processing), have very different requirements.
These workloads are so different that a single database cannot be optimal for both. A database fast for small transactions (OLTP) is too slow for big analytical scans (OLAP). This is why companies often use two separate systems: an operational database for running the business and a data warehouse for analyzing it.



##### Data Warehousing

To solve the problem of poor analytical performance on OLTP systems, companies use a separate system called a **Data Warehouse**. A data warehouse is a central repository of integrated data from one or more different sources, designed specifically for query and analysis (OLAP).

**Key Concepts:**

*   **ETL Process:** Data is moved into the warehouse via an **ETL** (Extract, Transform, Load) process:
    *   **Extract:** Pull data from operational databases (OLTP).
    *   **Transform:** Clean, normalize, and restructure the data to make it suitable for analysis.
    *   **Load:** Write the transformed data into the warehouse tables.

*   **Key Characteristics:**
    *   **Read-Only for Analysis:** Analysts query it; it's not for running the business.
    *   **Optimized for Scans:** Designed for large, complex analytical queries, not small, fast lookups.
    *   **Historical Data:** Maintains a history of changes over time, unlike OLTP systems which often just store the current state.

By separating the transactional (OLTP) and analytical (OLAP) workloads, a data warehouse allows for deep, complex analysis without impacting the performance of the day-to-day business operations.




##### Stars and Snowflakes: Schemas for Analytics


This section introduces two common design patterns for structuring data in a data warehouse: **Star Schema** and **Snowflake Schema**. The goal is to organize data in a way that makes analytical queries simple and fast.

**1. Star Schema**
This is the simplest and most common pattern. Its diagram resembles a star.
*   **Fact Table (Center):** Contains the numerical data of a business event (e.g., `sales_amount`, `quantity`). It also holds foreign keys to the dimension tables.
*   **Dimension Tables (Points):** Contain the descriptive context for the facts (e.g., `products`, `customers`, `dates`). They describe the "who, what, where, when" of an event.

**Key Characteristic:** It is **denormalized**, meaning dimension tables contain all necessary information. This minimizes the number of JOINs needed for a query, making it very fast.

**2. Snowflake Schema**
This is a more complex version of the star schema where dimension tables are **normalized**. For example, a `product` dimension might be split into a `product` table and a separate `category` table.

**Trade-off:** It reduces data redundancy and saves storage space, but it requires more JOINs, which makes queries slower and more complex.

The main difference is a trade-off between **query performance and storage size**. **Star Schema is generally preferred** in data warehousing. The small amount of space saved by a snowflake schema is not worth the cost of slower, more complex queries, as query performance and analyst simplicity are top priorities.





#### Column-Oriented Storage


This section introduces a fundamental design choice for analytical databases: **column-oriented storage**.

Instead of storing all data for a row together (row-oriented), columnar databases store all values for a single column together on disk.

*   **Example:** All `price` values are stored in one file, and all `product_name` values are stored in a separate file.

**Key Advantage for Analytics:**
For analytical queries (OLAP) that often only need a few columns (e.g., `SELECT SUM(price) FROM sales`), this design is vastly more efficient. The database only needs to read the `price` column from disk, completely ignoring the other columns. This dramatically reduces the amount of data read from disk, making analytical queries significantly faster than on row-oriented systems.


##### Column Compression


Column compression is extremely effective because data within a single column is often very similar or repetitive. This allows for highly efficient compression techniques like **Run-Length Encoding** (e.g., storing 'USA, USA, USA' as 'USAx3').

The primary benefit of compression in analytical databases is not just saving storage space, but **dramatically increasing query speed**.

This is because disk I/O is often the biggest bottleneck. Reading a small, compressed file from disk is much faster than reading a large, uncompressed one. The CPU cost of decompressing the data is trivial compared to the time saved on the slow disk operation.

In essence, column compression acts as a **performance accelerator**: it reduces the amount of data read from disk, allowing analytical queries to run significantly faster.


##### Sort Order in Column Storage


Sorting the data within each column file provides two major advantages for analytical queries.

1.  **Better Compression:** Sorting groups identical values together, which makes compression algorithms (like Run-Length Encoding) much more efficient, further reducing disk space usage.

2.  **Faster Queries:** When a column is sorted, the database can use a fast **binary search** to quickly locate a value or a range of values, instead of scanning the entire column. This makes filtering on the sorted column extremely fast.

**The Trade-off: The Sort Key**
You cannot sort all columns at once. You must choose one or more columns as the **sort key**. This is a strategic decision:
*   Queries that filter or aggregate on the **sort key** will be exceptionally fast.
*   Queries on other columns will not get this speed-up (though they still benefit from columnar compression).

In essence, choosing a sort key is a way to pre-optimize your storage for the most common and critical analytical queries.


##### Writing to Column-Oriented Storage

Writing a single row to a columnar database is inefficient because it requires updating multiple separate files on disk—one for each column.

To solve this, columnar systems use a batching strategy:
- **In-Memory Buffer:** Incoming writes are first stored in a fast in-memory buffer (a MemTable or Write Buffer).
Batch Write: When the buffer is full, the system performs a single, large write operation. It takes all the accumulated rows, separates them by column, and appends the new data to each column file in one go.
- **The Trade-off:**
This approach makes write throughput (total data written per second) very high. However, it introduces latency (a delay) for any single write, as it waits in the buffer.

This is why columnar storage is excellent for analytical workloads (OLAP), which often involve bulk data loads, but unsuitable for transactional workloads (OLTP) that require immediate, low-latency writes.




##### Aggregation: Data Cubes and Materialized Views

To avoid repeatedly running expensive aggregation queries (e.g., "total sales per product for last month"), systems use pre-computation.

1. Materialized Views
Unlike a regular view (a stored query), a materialized view stores the result of a query as a physical table. Queries reading from it are instant.

Trade-off: The data can become stale and must be periodically refreshed to stay in sync with the underlying tables.
2. Data Cubes
A more advanced, multi-dimensional version of a materialized view. Think of a cube with axes (dimensions) like Time, Product, and Location. Each cell contains a pre-aggregated value (e.g., total sales).

**Benefit:** Enables extremely fast "drill-down" and interactive analysis for business intelligence dashboards.
The Core Principle:
The fundamental idea is to trade storage space for query speed. By pre-calculating and storing results, systems can provide instant answers to complex analytical questions that would otherwise be very slow to compute.



#### Summary

This chapter explores the internal data structures that power databases and how they are optimized for different workloads.

The chapter begins by distinguishing between two primary workloads:
*   **OLTP (Online Transaction Processing):** Fast, small, real-time operations (e.g., user orders).
*   **OLAP (Online Analytical Processing):** Large, complex scans of historical data for analysis (e.g., monthly sales reports).

For **OLTP**, the chapter compares two fundamental storage structures:
*   **LSM-Trees:** Optimized for high write throughput. They append writes to an in-memory table and later flush them to disk as immutable, sorted **SSTables**. A background **compaction** process merges files to keep reads efficient.
*   **B-Trees:** Optimized for fast, predictable reads. They keep data sorted on disk at all times, allowing for efficient lookups and range queries. Writes are slower due to potential page splits. This is the most common index structure in relational databases.

The chapter then shifts to **OLAP**, explaining that analytical workloads are best handled by a separate **data warehouse** populated via an **ETL** process. The preferred data model here is the **Star Schema**, which denormalizes data for fast querying.

The core technology for modern data warehouses is **Column-Oriented Storage**. Instead of storing rows together, it stores all values for a single column together. This provides massive advantages for analytical queries:
*   **Fast Queries:** Queries only read the columns they need, drastically reducing I/O.
*   **High Compression:** Data within a column is uniform, allowing for excellent compression, which further speeds up queries.
*   **Sort Keys:** Sorting data within a column file enhances compression and enables extremely fast range queries on that column.

Finally, the chapter covers how writing works in columnar systems (batching writes in memory) and how aggregation is made instant using **materialized views** and **data cubes**, which trade storage space for query speed.


### CHAPTER 4

This chapter addresses a fundamental challenge in software engineering: data changes over time. As systems evolve, the structure of their data also evolves. The core problem is ensuring that different parts of a system—potentially running different versions of the code—can still communicate effectively.

**The Two Core Problems:**

**Data in Motion:** Data is constantly moving between processes (e.g., from a client to a server, or between microservices). Each process needs to decode the data sent by the other.

**Evolving Data:** The format of this data is not static. New fields are added, old ones are removed. This creates a critical need for compatibility.

**The Goal: Compatibility**

The chapter focuses on achieving two types of compatibility to prevent systems from breaking as they evolve:

- **Backward Compatibility:** New code can read data written by old code.
- **Forward Compatibility:** Old code can read data written by new code (by gracefully ignoring new fields it doesn't understand).


The chapter will explore different data encoding formats (like JSON, Protocol Buffers, Avro) and how well they support this kind of evolution, ensuring that our systems remain robust and maintainable over time.


#### Formats for Encoding Data
This section compares different ways to encode data in memory (objects) into a byte sequence for storage or network transmission, focusing on their ability to handle schema evolution.

##### 1. Language-Specific Formats (e.g., Java Serialization, Python Pickle)

Pros: Easy to use within a single language.
Cons: Language lock-in (cannot be read by other languages), major security risks, and very poor support for schema evolution. They are unsuitable for data that needs to be stored long-term or shared between systems.
##### 2. Standard Text Formats (JSON, XML)

Pros: Human-readable and widely supported, making them great for interoperability.
Cons: Verbose (take up more space) and ambiguous about data types (e.g., is a number an integer or a string?). They lack a formal schema. Binary variants like MessagePack reduce size but not the ambiguity.
##### 3. Binary Schema-Based Formats (Thrift, Protocol Buffers)

These require defining a schema first, which is then used to generate code for various languages.
Pros: Provide strong typing, compact binary encoding, and excellent performance.
Key Strength: Their key advantage is a well-defined set of rules for schema evolution, allowing for forward and backward compatibility (e.g., you can add new optional fields without breaking old code).










#### Modes of Dataflow

##### Dataflow Through Databases

##### Dataflow Through Services: REST and RPC

##### Message-Passing Dataflow







#### Summary


## PART II: Distributed Data



## PART III: Derived Data