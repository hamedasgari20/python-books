# Architecture Patterns with Python: Enabling Test-Driven Development, Domain-Driven Design, and Event-Driven Microservices

Authors: Bob Gregory, Harry Percival

![](images/book_first_page.png)

<!-- TOC -->
* [Architecture Patterns with Python: Enabling Test-Driven Development, Domain-Driven Design, and Event-Driven Microservices](#architecture-patterns-with-python-enabling-test-driven-development-domain-driven-design-and-event-driven-microservices)
  * [Introduction](#introduction)
  * [Chapter1: Domain Modeling](#chapter1-domain-modeling)
  * [Chapter2: Repository Pattern](#chapter2-repository-pattern)
  * [Chapter3: A Brief Interlude: On Coupling and Abstractions](#chapter3-a-brief-interlude-on-coupling-and-abstractions)
  * [Chapter4: Our First Use Case: Flask API and Service Layer](#chapter4-our-first-use-case-flask-api-and-service-layer)
  * [Chapter5: TDD in High Gear and Low Gear](#chapter5-tdd-in-high-gear-and-low-gear)
  * [Chapter6: Unit of Work Pattern](#chapter6-unit-of-work-pattern)
  * [Chapter7: Aggregates and Consistency Boundaries](#chapter7-aggregates-and-consistency-boundaries)
* [Event-Driven Architecture](#event-driven-architecture)
<!-- TOC -->

[The source code in GitHub](https://github.com/cosmicpython/code)

## Introduction

Software systems, tend toward chaos. When we first start building a new system,
we have grand ideas that our code will be clean and well ordered, but over time we
find that it ends up a number of confusing classes and modules. This is so common that software engineers have
their own term for chaos: the Big Ball of Mud anti-pattern.

![](images/big_ball_of_mud.png "Big ball of mud")

If you’d like a picture of where we’re going, take a look at Figure below, but don’t worry if
none of it makes sense yet! We introduce each box in the figure, one by one.

![](images/component_diagram_for_our_app.png)

## Chapter1: Domain Modeling
This chapter looks into how we can model business processes with code, in a way
that’s highly compatible with TDD.

- **What Is a Domain Model?**

The **domain** is a fancy way of saying the problem you’re trying to solve. Depending on which system you’re
talking about, the domain might be purchasing and procurement, or product design,
or logistics and delivery. A **model** is a map of a process or phenomenon that captures a useful property. The domain model is the mental map that business owners have of their businesses.

In a nutshell, DDD says that the most important thing about software is that it provides a useful model of a problem. If we get that model right, our software delivers
value and makes new things possible.

- **Problem definition**

We’re going to use a real-world domain model throughout this book, specifically a
model from our current employment. MADE.com is a successful furniture retailer.
We source our furniture from manufacturers all over the world and sell it across
Europe. When you buy a sofa or a coffee table, we have to figure out how best to get your
goods from Poland or China or Vietnam and into your living room. At a high level, we have separate systems that are responsible for buying stock, selling
stock to customers, and shipping goods to customers. A system in the middle needs
to coordinate the process by allocating stock to a customer’s orders.

![](images/diagram_of_the_company.png)

For the purposes of this book, we’re imagining that the business decides to implement
an exciting new way of allocating stock. Until now, the business has been presenting
stock and lead times based on what is physically available in the warehouse. If and
when the warehouse runs out, a product is listed as “out of stock” until the next shipment arrives from the manufacturer.

Here’s the innovation: if we have a system that can keep track of all our shipments
and when they’re due to arrive, we can treat the goods on those ships as real stock and
part of our inventory, just with slightly longer lead times. Fewer goods will appear to
be out of stock, we’ll sell more, and the business can save money by keeping lower
inventory in the domestic warehouse. We need a more complex allocation mechanism. **Time for
some domain modeling.**


- **Exploring the Domain Language**

The following notes we might have taken while having a conversation with our domain experts about allocation.

1- A product is identified by a SKU, pronounced “skew,” which is short for stock-keeping
unit. Customers place orders. An order is identified by an order reference and comprises multiple order lines, where each line has a SKU and a quantity. For example: _10 units of RED-CHAIR_

2- The purchasing department orders small batches of stock. A batch of stock has a
unique ID called a reference, a SKU, and a quantity.

3- We need to allocate order lines to batches. When we’ve allocated an order line to a
batch, we will send stock from that specific batch to the customer’s delivery address

4- We can’t allocate to a batch if the available quantity is less than the quantity of the
order line

5- We can’t allocate the same line twice.

- **Unit Testing Domain Models**

You’ll find some placeholder unit tests on [GitHub](https://github.com/cosmicpython/code/blob/chapter_01_domain_model_exercise/test_model.py), but you could just start from scratch, or combine/rewrite them however you like.

here is a sample unit test:

```angular2html
def test_allocating_to_a_batch_reduces_the_available_quantity():
     batch = Batch("batch-001", "SMALL-TABLE", qty=20, eta=date.today())
     line = OrderLine('order-ref', "SMALL-TABLE", 2)
     batch.allocate(line)
     assert batch.available_quantity == 18
```

The name of our unit test describes the behavior that we want to see from the system,
and the names of the classes and variables that we use are taken from the business
jargon. And here is a domain model that meets our requirements:

here is our UML and domain model:

![](images/uml.png)

todo: explain _purchased_quantity and allocations with more details

```angular2html

@dataclass(frozen=True)
class OrderLine:
    sku: str
    qty: int


class Batch:
    def __init__(self, ref: str, sku: str, qty: int, eta: Optional[date]):
        self.reference = ref
        self.sku = sku
        self.eta = eta
        self._purchased_quantity = qty
        self._allocations = set() 


    def allocate(self, line: OrderLine):
        if self.can_allocate(line):
            self._allocations.add(line)

    def deallocate(self, line: OrderLine):
        if line in self._allocations:
            self._allocations.remove(line)

    @property
    def allocated_quantity(self) -> int:
        return sum(line.qty for line in self._allocations)

    @property
    def available_quantity(self) -> int:
        return self._purchased_quantity - self.allocated_quantity

    def can_allocate(self, line: OrderLine) -> bool:
        return self.sku == line.sku and self.available_quantity >= line.qty

```
Now we’re getting somewhere! A batch now keeps track of a set of allocated Order
Line objects. When we allocate, if we have enough available quantity, we just add to
the set. Our available_quantity is now a calculated property: purchased quantity
minus allocated quantity.

- **Value Object**

Whenever we have a business concept that has data but no identity, we often choose
to represent it using the **Value Object** pattern. A value object is any domain object that
is uniquely identified by the data it holds. in previous example **OrderLine** is value object:

```angular2html

@dataclass(frozen=True)
class OrderLine:
    sku: str
    qty: int

```

todo: Why OrderLine is value object? How can we save its related data in DB?

- **Entities**

We use the term entity to describe a domain object that has long-lived identity. We can change their values, and they are
still recognizably the same thing. Batches, in our example, are entities.

- **Not Everything Has to Be an Object: A Domain Service Function**

A thing that allocates an order line, given a set of batches, sounds a lot like a function, and we can take advantage of the fact that Python is a multiparadigm language and just make it a function.
for example allocate service can act as the below:

```angular2html
def allocate(line: OrderLine, batches: List[Batch]) -> str:
    try:
        batch = next(b for b in sorted(batches) if b.can_allocate(line))
        batch.allocate(line)
        return batch.reference
    except StopIteration:
        raise OutOfStock(f"Out of stock for sku {line.sku}")
```

Python is a multiparadigm language, so let the “verbs” in your code be functions.

Figure below is a visual representation of where we’ve ended up.

![](images/domain_model_in_first_chapter.png)

That’ll probably do for now! We have a domain service that we can use for our first
use case. 

## Chapter2: Repository Pattern
It’s time to use the dependency inversion principle as a way of decoupling our core logic from infrastructural concerns. 
We’ll introduce the **Repository pattern**, a simplifying abstraction over data storage, allowing us to decouple our model layer from the business layer.
Figure below shows a little preview of what we’re going to build: a Repository object that sits between our domain model and the database.

![](images/before_and_after_the_repository_pattern.png)

The code for this chapter is in the [chapter_02_repository branch](https://github.com/cosmicpython/code/tree/chapter_02_repository)

When we build our first API endpoint, we know we’re going to have some code that looks more or less like the following.

```angular2html
@flask.route.gubbins
def allocate_endpoint():
     # extract order line from request
     line = OrderLine(request.params, ...)
     # load all batches from the DB
     batches = ...
     # call our domain service
     allocate(line, batches)
     # then save the allocation back to the database somehow
     return 201
```
At this point, though, our API endpoint might look something like the following, and we could get it to work just fine:
Using SQLAlchemy directly in our API endpoint

```angular2html
@flask.route.gubbins
def allocate_endpoint():
     session = start_session()
     # extract order line from request
     line = OrderLine(
     request.json['orderid'],
     request.json['sku'],
     request.json['qty'],
     )
     # load all batches from the DB
     batches = session.query(Batch).all()
     # call our domain service
     allocate(line, batches)
     # save the allocation back to the database
     session.commit()
     return 201
```

- **Introducing the Repository Pattern**

The **Repository Pattern** is a design pattern that acts as an abstraction layer between the data access layer and the business logic of an application. It encapsulates all data access logic and provides a clean API for the rest of the application to interact with, allowing for better separation of concerns and easier testing.
Here’s what an abstract base class (ABC) for our repository would look like:

```angular2html
class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, batch: model.Batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, reference) -> model.Batch:
        raise NotImplementedError
```
And here is a typical repository:

```angular2html

class SqlAlchemyRepository(AbstractRepository):
    def __init__(self, session):
        self.session = session

    def add(self, batch):
        self.session.add(batch)

    def get(self, reference):
        return self.session.query(model.Batch).filter_by(reference=reference).one()

    def list(self):
        return self.session.query(model.Batch).all()
```
And now our Flask endpoint might look something like the following:

```angular2html
@flask.route.gubbins
def allocate_endpoint():
     batches = SqlAlchemyRepository.list()
     lines = [
     OrderLine(l['orderid'], l['sku'], l['qty'])
     for l in request.params...
     ]
     allocate(lines, batches)
     session.commit()
     return 201
```

todo: What is the advantage of the mentioned approach? Explain its functionality

 - **What Is a Port and What Is an Adapter, in Python?** (Perplexity is used for better definition in this section)

In the context of the ports and adapters architectural pattern, a port is an abstract interface that defines how the application interacts with external components, while an adapter is a concrete implementation of that interface that handles the communication with the external component.

A **port** is an abstract interface, usually defined as an abstract base class (ABC) or a protocol, that specifies the methods and attributes that adapters must implement. For example:

```angular2html
from abc import ABC, abstractmethod

class PrintAdapterPort(ABC):
    @abstractmethod
    def print_number(self, number: int):
        raise NotImplementedError
```

In this example, **PrintAdapterPort** is a port that defines a **print_number** method for printing an integer. 

An **adapter** is a concrete class that implements a port's interface and handles the communication with an external component. For example:

```angular2html
class ConsolePrintAdapter(PrintAdapterPort):
    def print_number(self, number: int):
        print(f"The generated number is: {number}")
```

Here, **ConsolePrintAdapter** is an adapter that implements the **PrintAdapterPort** interface by printing the number to the console.

## Chapter3: A Brief Interlude: On Coupling and Abstractions

Allow us a brief digression on the subject of abstractions, dear reader.
The code for this chapter is in the [chapter_03_abstractions branch](https://github.com/cosmicpython/code/tree/chapter_03_abstractions)

A key theme in this book, hidden among the fancy patterns, is that we can use simple
abstractions to hide messy details. When we’re unable to change component A for fear of breaking component B, we say
that the components have become **coupled**. Globally, coupling is a nuisance: it increases the risk and the cost of changing our
code, sometimes to the point where we feel unable to make any changes at all.
We can reduce the degree of coupling within a system by **abstracting
away the details**.

Let’s see an example. Imagine we want to write code for synchronizing two file directories, which we’ll call the **_source_** and the **_destination_**:

1- If a file exists in the source but not in the destination, copy the file over.

2- If a file exists in the source, but it has a different name than in the destination, rename the destination file to match.

3- If a file exists in the destination but not in the source, remove it.

Our first and third requirements are simple enough: we can just compare two lists of paths. Our second is trickier, though. To detect renames, we’ll have to inspect the
content of files. For this, we can use a hashing function like MD5 or SHA-1. The code to generate a SHA-1 hash from a file is simple enough:

```angular2html
def hash_file(path):
    hasher = hashlib.sha1()
    with path.open("rb") as file:
        buf = file.read(BLOCKSIZE)
        while buf:
            hasher.update(buf)
            buf = file.read(BLOCKSIZE)
    return hasher.hexdigest()
```

todo: explain more this method

Our first hackish approach looks something like this:

```angular2html
import hashlib
import os
import shutil
from pathlib import Path

def sync(source, dest):
    # Walk the source folder and build a dict of filenames and their hashes
   source_hashes = {}
   for folder, _, files in os.walk(source):
     for fn in files:
       source_hashes[hash_file(Path(folder) / fn)] = fn

   seen = set() # Keep track of the files we've found in the target
   # Walk the target folder and get the filenames and hashes
   for folder, _, files in os.walk(dest):
     for fn in files:
       dest_path = Path(folder) / fn
       dest_hash = hash_file(dest_path)
       seen.add(dest_hash)
       # if there's a file in target that's not in source, delete it
       if dest_hash not in source_hashes:
         dest_path.remove()
       # if there's a file in target that has a different path in source,
       # move it to the correct path
       elif dest_hash in source_hashes and fn != source_hashes[dest_hash]:
         shutil.move(dest_path, Path(folder) / source_hashes[dest_hash])

   # for every file that appears in source but not target, copy the file to
   # the target
   for src_hash, fn in source_hashes.items():
     if src_hash not in seen:
       shutil.copy(Path(source) / fn, Path(dest) / fn)
```
todo: explain more this method


Fantastic! We have some code and it looks OK. The problem is that our domain
logic, “figure out the difference between two directories,” is tightly coupled to the I/O
code. We can’t run our difference algorithm without calling the pathlib, shutil, and
hashlib modules. On top of that, our code isn’t very extensible. Our high-level code is coupled to low-level details, and it’s making life hard.

- **What could we do to rewrite our code to make it more testable?**

We can think of these as three distinct responsibilities that the code has:

1. We interrogate the filesystem by using os.walk and determine hashes for a series
of paths. This is similar in both the source and the destination cases.
2. We decide whether a file is new, renamed, or redundant.
3. We copy, move, or delete files to match the source.


- **Implementing Our Chosen Abstractions**

Here is refactored version of my codes:

```angular2html
def sync(source, dest):
    # imperative shell step 1, gather inputs
    source_hashes = read_paths_and_hashes(source)
    dest_hashes = read_paths_and_hashes(dest)

    # step 2: call functional core
    actions = determine_actions(source_hashes, dest_hashes, source, dest)

    # imperative shell step 3, apply outputs
    for action, *paths in actions:
        if action == "COPY":
            shutil.copyfile(*paths)
        if action == "MOVE":
            shutil.move(*paths)
        if action == "DELETE":
            os.remove(paths[0])

def hash_file(path):
    hasher = hashlib.sha1()
    with path.open("rb") as file:
        buf = file.read(BLOCKSIZE)
        while buf:
            hasher.update(buf)
            buf = file.read(BLOCKSIZE)
    return hasher.hexdigest()


def read_paths_and_hashes(root):
    hashes = {}
    for folder, _, files in os.walk(root):
        for fn in files:
            hashes[hash_file(Path(folder) / fn)] = fn
    return hashes


def determine_actions(source_hashes, dest_hashes, source_folder, dest_folder):
    for sha, filename in source_hashes.items():
        if sha not in dest_hashes:
            sourcepath = Path(source_folder) / filename
            destpath = Path(dest_folder) / filename
            yield "COPY", sourcepath, destpath

        elif dest_hashes[sha] != filename:
            olddestpath = Path(dest_folder) / dest_hashes[sha]
            newdestpath = Path(dest_folder) / filename
            yield "MOVE", olddestpath, newdestpath

    for sha, filename in dest_hashes.items():
        if sha not in source_hashes:
            yield "DELETE", dest_folder / filename
```
The **determine_actions()** function will be the core of our business logic, which says,
“Given these two sets of hashes and filenames, what should we copy/move/delete?”. It
takes simple data structures and returns simple data structures:

todo: Explain this method with more details

## Chapter4: Our First Use Case: Flask API and Service Layer

In this chapter, we discuss the differences between orchestration logic, business logic,
and interfacing code, and we introduce the Service Layer pattern to take care of
orchestrating our workflows and defining the use cases of our system.

Figure below shows what we’re aiming for: we’re going to add a Flask API that will talk
to the service layer, which will serve as the entrypoint to our domain model. Because
our service layer depends on the **AbstractRepository**, we can unit test it by using
**FakeRepository** but run our production code using **SqlAlchemyRepository**.

todo: Where is FakeRepository come from?

![](images/service_layer_added.png)

The code for this chapter is in the [chapter_04_service_layer](https://github.com/cosmicpython/code/blob/chapter_04_service_layer/services.py)

- **A First End-to-End Test**

For now, we want to write one or maybe two tests that are going to exercise a “real”
API endpoint (using HTTP) and talk to a real database. Let’s call them end-to-end
tests because it’s one of the most self-explanatory names.

```angular2html
@pytest.mark.usefixtures("restart_api")
def test_happy_path_returns_201_and_allocated_batch(add_stock):
    sku, othersku = random_sku(), random_sku("other")
    earlybatch = random_batchref(1)
    laterbatch = random_batchref(2)
    otherbatch = random_batchref(3)
    add_stock(
        [
            (laterbatch, sku, 100, "2011-01-02"),
            (earlybatch, sku, 100, "2011-01-01"),
            (otherbatch, othersku, 100, None),
        ]
    )
    data = {"orderid": random_orderid(), "sku": sku, "qty": 3}
    url = config.get_api_url()

    r = requests.post(f"{url}/allocate", json=data)

    assert r.status_code == 201
    assert r.json()["batchref"] == earlybatch
```

**random_sku()**, **random_batchref()**, and so on are little helper functions that
generate randomized characters by using the uuid module. **add_stock** is a helper fixture that just hides away the details of manually inserting rows into the database using SQL.

- **The Straightforward Implementation**

Implementing things in the most obvious way, you might get something like this:

```angular2html
@app.route("/allocate", methods=['POST'])
def allocate_endpoint():
   session = get_session()
   batches = repository.SqlAlchemyRepository(session).list()
   line = model.OrderLine(
   request.json['orderid'],
   request.json['sku'],
   request.json['qty'],
   )
   batchref = model.allocate(line, batches)
   return jsonify({'batchref': batchref}), 201
```

But if we want to add more error handling to this API it should be changed as below:

```angular2html
def is_valid_sku(sku, batches):
    return sku in {b.sku for b in batches}

@app.route("/allocate", methods=['POST'])
def allocate_endpoint():
   session = get_session()
   batches = repository.SqlAlchemyRepository(session).list()
   line = model.OrderLine(
     request.json['orderid'],
     request.json['sku'],
     request.json['qty'],
   )
   if not is_valid_sku(line.sku, batches):
        return jsonify({'message': f'Invalid sku {line.sku}'}), 400
   try:
        batchref = model.allocate(line, batches)
   except model.OutOfStock as e:
        return jsonify({'message': str(e)}), 400
   session.commit()
   return jsonify({'batchref': batchref}), 201

```
But our Flask app is starting to look a bit unwieldy.

- **Introducing a Service Layer, and Using FakeRepository to Unit Test It**

If we look at what our Flask app is doing, there’s quite a lot of what we might call
orchestration—fetching stuff out of our repository, validating our input against data‐
base state, handling errors, and committing in the happy path. In the following we add service layer in TDD manner.

Our fake repository, an in-memory collection of batches is as below:

```angular2html
class FakeRepository(repository.AbstractRepository):
    def __init__(self, batches):
        self._batches = set(batches)

    def add(self, batch):
        self._batches.add(batch)

    def get(self, reference):
        return next(b for b in self._batches if b.reference == reference)

    def list(self):
        return list(self._batches)
```
FakeRepository holds the Batch objects that will be used by our test. Unit testing with fakes at the service layer:

```angular2html
def test_returns_allocation():
    line = model.OrderLine("o1", "COMPLICATED-LAMP", 10)
    batch = model.Batch("b1", "COMPLICATED-LAMP", 100, eta=None)
    repo = FakeRepository([batch])

    result = services.allocate(line, repo, FakeSession())
    assert result == "b1"


def test_error_for_invalid_sku():
    line = model.OrderLine("o1", "NONEXISTENTSKU", 10)
    batch = model.Batch("b1", "AREALSKU", 100, eta=None)
    repo = FakeRepository([batch])

    with pytest.raises(services.InvalidSku, match="Invalid sku NONEXISTENTSKU"):
        services.allocate(line, repo, FakeSession())
```

And finally we’ll write a service function that looks something like this (services.py):

```angular2html
class InvalidSku(Exception):
    pass


def is_valid_sku(sku, batches):
    return sku in {b.sku for b in batches}


def allocate(line: OrderLine, repo: AbstractRepository, session) -> str:
    batches = repo.list()
    if not is_valid_sku(line.sku, batches):
        raise InvalidSku(f"Invalid sku {line.sku}")
    batchref = model.allocate(line, batches)
    session.commit()
    return batchref
```
Our services module (services.py) will define an **allocate()** service-layer function. It will sit between our API layer and
the **allocate()** domain service function from our domain model. Here We fetch some objects from the repository. We make some checks or assertions about the request against the current state of
the world. We call a domain service and If all is well, we save/update any state we’ve changed. That last step is a little unsatisfactory at the moment, as our service layer is tightly
coupled to our database layer. We’ll improve that in Chapter 6 with the **Unit of Work pattern**.

- **Depend on Abstractions**

Notice one more thing about our service-layer function:
It depends on a repository. We’ve chosen to make the dependency explicit, and we’ve
used the type hint to say that we depend on **AbstractRepository**. This means it’ll
work both when the tests give it a **FakeRepository** and when the Flask app gives it a
**SqlAlchemyRepository**. If you remember **“The Dependency Inversion Principle”** this is what we
mean when we say we should “**depend on abstractions**.” Our high-level module, the
**service layer**, depends on the **repository abstraction**.

our Flask app now looks a lot
cleaner:
```angular2html
@app.route("/allocate", methods=["POST"])
def allocate_endpoint():
    session = get_session()
    repo = repository.SqlAlchemyRepository(session)
    line = model.OrderLine(
        request.json["orderid"], request.json["sku"], request.json["qty"],
    )

    try:
        batchref = services.allocate(line, repo, session)
    except (model.OutOfStock, services.InvalidSku) as e:
        return {"message": str(e)}, 400

    return {"batchref": batchref}, 201
```

We instantiate a database session and some repository objects. We extract the user’s commands from the web request and pass them to a
domain service. We return some JSON responses with the appropriate status codes.

The responsibilities of the Flask app are just standard web stuff: per-request session
management, parsing information out of POST parameters, response status codes,
and JSON. All the orchestration logic is in the use case/service layer, and the domain
logic stays in the domain.

- **Putting Things in Folders to See Where It All Belongs**
As our application gets bigger, we’ll need to keep tidying our directory structure. The
layout of our project gives us useful hints about what kinds of object we’ll find in each
file. Here’s one way we could organize things:

![](images/folders_structures.png)

1- Let’s have a folder for our domain model. Currently that’s just one file, but for a
more complex application, you might have one file per class; you might have
helper parent classes for **Entity**, **ValueObject**, and **Aggregate**, and you might
add an **exceptions.py** for domain-layer exceptions and, as you’ll see in Part II,
**commands.py** and **events.py**.

2- We’ll distinguish the service layer. Currently that’s just one file called **services.py**
for our service-layer functions. You could add service-layer exceptions here, and
as you’ll see in Chapter 5, we’ll add **unit_of_work.py**.

3- Adapters is a nod to the ports and adapters terminology. This will fill up with any
other abstractions around external I/O (e.g., a redis_client.py). Strictly speaking,
you would call these secondary adapters or driven adapters, or sometimes inwardfacing adapters.

4- Entrypoints are the places we drive our application from. In the official ports and
adapters terminology, these are adapters too, and are referred to as primary, driv‐
ing, or outward-facing adapters.

- **The DIP in Action**

Here are the dependencies tree in our app when we run tests and when we run the application.

![](images/add_dependency.png)
![](images/tests_dependency.png)
![](images/runtime_dependency.png)

todo: explain more details about these figures and DIP

But there are still some bits of awkwardness to tidy up:

1- The service layer is still tightly coupled to the domain, because its API is
expressed in terms of OrderLine objects. In Chapter 5, we’ll fix that and talk
about the way that the service layer enables more productive TDD.

2- The service layer is tightly coupled to a session object. In Chapter 6, we’ll introduce one more pattern that works closely with the Repository and Service Layer
patterns, the Unit of Work pattern, and everything will be absolutely lovely. You’ll
see!

## Chapter5: TDD in High Gear and Low Gear
[Source Code](https://github.com/cosmicpython/code/tree/chapter_05_high_gear_low_gear)

We’ve introduced the service layer to capture some of the additional orchestration
responsibilities we need from a working application. The service layer helps us clearly
define our use cases and the workflow for each: what we need to get from our repositories, what pre-checks and current state validation we should do, and what we save at
the end. But currently, many of our unit tests operate at a lower level, acting directly on the
model. In this chapter we’ll discuss the trade-offs involved in moving those tests up to
the service-layer level, and some more general testing guidelines.

- **Should Domain Layer Tests Move to the Service Layer?**

Since we can test our software against the service layer, we don’t really need tests for the domain model anymore.
Instead, we could rewrite all of the domain-level tests from Chapter 1 in terms of the service layer. here is an example:

domain-layer test:

```angular2html
def test_prefers_current_stock_batches_to_shipments():
   in_stock_batch = Batch("in-stock-batch", "RETRO-CLOCK", 100, eta=None)
   shipment_batch = Batch("shipment-batch", "RETRO-CLOCK", 100, eta=tomorrow)
   line = OrderLine("oref", "RETRO-CLOCK", 10)

   allocate(line, [in_stock_batch, shipment_batch])

   assert in_stock_batch.available_quantity == 90
   assert shipment_batch.available_quantity == 100

```
service-layer test:

```angular2html
def test_prefers_warehouse_batches_to_shipments():
   in_stock_batch = Batch("in-stock-batch", "RETRO-CLOCK", 100, eta=None)
   shipment_batch = Batch("shipment-batch", "RETRO-CLOCK", 100, eta=tomorrow)
   repo = FakeRepository([in_stock_batch, shipment_batch])
   session = FakeSession()
   line = OrderLine('oref', "RETRO-CLOCK", 10)

  services.allocate(line, repo, session)

  assert in_stock_batch.available_quantity == 90
  assert shipment_batch.available_quantity == 100

```

Tests are supposed to help us change our system fearlessly, but often we see teams
writing too many tests against their domain model. This causes problems when they
come to change their codebase and find that they need to update tens or even hundreds of unit tests.

As we get further into the book, you’ll see how the service layer forms an API for our
system that we can drive in multiple ways. Testing against this API reduces the
amount of code that we need to change when we refactor our domain model. If we
restrict ourselves to testing only against the service layer, we won’t have any tests that
directly interact with “private” methods or attributes on our model objects, which
leaves us freer to refactor them.

- **On Deciding What Kind of Tests to Write**

You might be asking yourself, “Should I rewrite all my unit tests, then? Is it wrong to
write tests against the domain model?” To answer those questions, it’s important to
understand the trade-off between coupling and design feedback (see Figure below).

![](images/test_spectrum.png)

We only get that feedback, though, when we’re working closely with the target code.
A test for the HTTP API tells us nothing about the fine-grained design of our objects,
because it sits at a much higher level of abstraction. On the other hand, we can rewrite our entire application and, so long as we don’t
change the URLs or request formats, our HTTP tests will continue to pass. This gives
us confidence that large-scale changes, like changing the database schema, haven’t
broken our code.


- **High and Low Gear**

When starting a new project or when hitting a particularly gnarly problem, we will
drop back down to writing tests against the domain model so we get better feedback
and executable documentation of our intent.

The metaphor we use is that of shifting gears. When starting a journey, the bicycle
needs to be in a low gear so that it can overcome inertia. Once we’re off and running,
we can go faster and more efficiently by changing into a high gear; but if we suddenly
encounter a steep hill or are forced to slow down by a hazard, we again drop down to
a low gear until we can pick up speed again.

- **Fully Decoupling the Service-Layer Tests from the Domain**

We still have direct dependencies on the domain in our service-layer tests, because we
use domain objects to set up our test data and to invoke our service-layer functions.
To have a service layer that’s fully decoupled from the domain, we need to rewrite its
API to work in terms of primitives.
Our service layer currently takes an OrderLine domain object:

```angular2html
def allocate(line: OrderLine, repo: AbstractRepository, session) -> str:
```
How would it look if its parameters were all primitive types?

```angular2html
def allocate(orderid: str, sku: str, qty: int, repo: AbstractRepository, session) -> str:
```

Tests now use primitives in function call

```angular2html
def test_returns_allocation():
   batch = model.Batch("batch1", "COMPLICATED-LAMP", 100, eta=None)
   repo = FakeRepository([batch])

   result = services.allocate("o1", "COMPLICATED-LAMP", 10, repo, FakeSession())
   assert result == "batch1"

```
But our tests still depend on the domain, because we still manually instantiate **Batch**
objects. So, if one day we decide to massively refactor how our Batch model works,
we’ll have to change a bunch of tests.

- **Adding a Missing Service**

We could go one step further, though. If we had a service to add stock, we could use
that and make our service-layer tests fully expressed in terms of the service layer’s
official use cases, removing all dependencies on the domain:

```angular2html
def add_batch(
        ref: str, sku: str, qty: int, eta: Optional[date],
        repo: AbstractRepository, session):
     repo.add(model.Batch(ref, sku, qty, eta))
     session.commit()
```
Should you write a new service just because it would help remove
dependencies from your tests? Probably not. But in this case, we
almost definitely would need an **add_batch** service one day
anyway.

That now allows us to rewrite all of our service-layer tests purely in terms of the services themselves, using only primitives, and without any dependencies on the model:

```angular2html

def test_allocate_returns_allocation():
    repo, session = FakeRepository([]), FakeSession()
    services.add_batch("batch1", "COMPLICATED-LAMP", 100, None, repo, session)
    result = services.allocate("o1", "COMPLICATED-LAMP", 10, repo, session)
    assert result == "batch1"

def test_allocate_errors_for_invalid_sku():
    repo, session = FakeRepository([]), FakeSession()
    services.add_batch("b1", "AREALSKU", 100, None, repo, session)

    with pytest.raises(services.InvalidSku, match="Invalid sku NONEXISTENTSKU"):
        services.allocate("o1", "NONEXISTENTSKU", 10, repo, FakeSession())

```

This is a really nice place to be in. Our service-layer tests depend on only the service
layer itself, leaving us completely free to refactor the model as we see fit.

- **Carrying the Improvement Through to the E2E Tests**

In the same way that adding **add_batch** helped decouple our service-layer tests from
the model, adding an API endpoint to add a batch would remove the need for the
ugly add_stock fixture, and our E2E tests could be free of those hardcoded SQL queries and the direct dependency on the database

API for adding a batch:

```angular2html
@app.route("/add_batch", methods=["POST"])
def add_batch():
    session = get_session()
    repo = repository.SqlAlchemyRepository(session)
    eta = request.json["eta"]
    if eta is not None:
        eta = datetime.fromisoformat(eta).date()
    services.add_batch(
        request.json["ref"],
        request.json["sku"],
        request.json["qty"],
        eta,
        repo,
        session,
    )
    return "OK", 201
```

Now the API test would be as below:

```angular2html
def post_to_add_batch(ref, sku, qty, eta):
    url = config.get_api_url()
    r = requests.post(
        f"{url}/add_batch", json={"ref": ref, "sku": sku, "qty": qty, "eta": eta}
    )
    assert r.status_code == 201


@pytest.mark.usefixtures("postgres_db")
@pytest.mark.usefixtures("restart_api")
def test_happy_path_returns_201_and_allocated_batch():
    sku, othersku = random_sku(), random_sku("other")
    earlybatch = random_batchref(1)
    laterbatch = random_batchref(2)
    otherbatch = random_batchref(3)
    post_to_add_batch(laterbatch, sku, 100, "2011-01-02")
    post_to_add_batch(earlybatch, sku, 100, "2011-01-01")
    post_to_add_batch(otherbatch, othersku, 100, None)
    data = {"orderid": random_orderid(), "sku": sku, "qty": 3}

    url = config.get_api_url()
    r = requests.post(f"{url}/allocate", json=data)

    assert r.status_code == 201
    assert r.json()["batchref"] == earlybatch

```

- **Recap:**

1- Aim for one end-to-end test per feature

2- Write the bulk of your tests against the service layer

3- Maintain a small core of tests written against your domain mode

4- Error handling counts as a feature

5- Express your service layer in terms of primitives rather than domain objects.


## Chapter6: Unit of Work Pattern
[Source Code](https://github.com/cosmicpython/code/tree/chapter_06_uow)

In this chapter we’ll introduce the final piece of the puzzle that ties together the
Repository and Service Layer patterns: the **Unit of Work pattern**.

If the **Repository pattern** is our abstraction over the idea of persistent storage, the
**Unit of Work (UoW)** pattern is our abstraction over the idea of atomic operations. It
will allow us to finally and fully decouple our service layer from the data layer.

Figure below shows that, currently, a lot of communication occurs across the layers of
our infrastructure: the API talks directly to the database layer to start a session, it
talks to the repository layer to initialize SQLAlchemyRepository, and it talks to the
service layer to ask it to allocate. **Without UoW: API talks directly to three layers**

![](images/without_uow.png)

Figure below shows our target state. The Flask API now does only two things: it initializes a unit of work, and it invokes a service. The service collaborates with the UoW, but neither the service
function itself nor Flask now needs to talk directly to the database. And we’ll do it all using a lovely piece of Python syntax, a context manager.

![](images/with_uow.png)

- **The Unit of Work Collaborates with the Repository**

Here’s how the service layer will look when we’re finished:

```angular2html
def allocate(
    orderid: str, sku: str, qty: int,
    uow: unit_of_work.AbstractUnitOfWork) -> str:
    line = OrderLine(orderid, sku, qty)
    with uow:
        batches = uow.batches.list()
        if not is_valid_sku(line.sku, batches):
            raise InvalidSku(f"Invalid sku {line.sku}")
        batchref = model.allocate(line, batches)
        uow.commit()
    return batchref

```
We’ll start a UoW as a context manager. **uow.batches** is the batches repo, so the UoW provides us access to our permanent storage.
When we’re done, we commit or roll back our work, using the **UoW**.

- **Test-Driving a UoW with Integration Tests**

Here are our integration tests for the UOW:

```angular2html
def test_uow_can_retrieve_a_batch_and_allocate_to_it(session_factory):
    session = session_factory()
    insert_batch(session, "batch1", "HIPSTER-WORKBENCH", 100, None)
    session.commit()

    uow = unit_of_work.SqlAlchemyUnitOfWork(session_factory)
    with uow:
        batch = uow.batches.get(reference="batch1")
        line = model.OrderLine("o1", "HIPSTER-WORKBENCH", 10)
        batch.allocate(line)
        uow.commit()

    batchref = get_allocated_batch_ref(session, "o1", "HIPSTER-WORKBENCH")
    assert batchref == "batch1"
```
We initialize the UoW by using our custom session factory and get back a uow
object to use in our with block. The UoW gives us access to the batches repository via **uow.batches**. We call **commit()** on it when we’re done.

- **Unit of Work and Its Context Manager**

Abstract UoW context manager:

```angular2html
class AbstractUnitOfWork(abc.ABC):
    batches: repository.AbstractRepository

    def __enter__(self) -> AbstractUnitOfWork:
        return self

    def __exit__(self, *args):
        self.rollback()

    @abc.abstractmethod
    def commit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def rollback(self):
        raise NotImplementedError
```
The UoW provides an attribute called **batches**, which will give us access to the
batches repository. If you’ve never seen a context manager, __enter__ and __exit__ are the two
magic methods that execute when we enter the with block and when we exit it,
respectively. They’re our setup and teardown phases. We’ll call this method to explicitly commit our work when we’re ready.
If we don’t commit, or if we exit the context manager by raising an error, we do a **rollback**.

Here is SQLAlchemy UoW:
```angular2html
class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, session_factory=DEFAULT_SESSION_FACTORY):
        self.session_factory = session_factory

    def __enter__(self):
        self.session = self.session_factory()  # type: Session
        self.batches = repository.SqlAlchemyRepository(self.session)
        return super().__enter__()

    def __exit__(self, *args):
        super().__exit__(*args)
        self.session.close()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()
```
The module defines a default session factory that will connect to Postgres, but we
allow that to be overridden in our integration tests so that we can use SQLite
instead. The __enter__ method is responsible for starting a database session and instantiating a real repository that can use that session.
We close the session on exit. Finally, we provide concrete **commit()** and **rollback()** methods that use our
database session.

- **Fake Unit of Work for Testing**

Here’s how we use a fake UoW in our service-layer tests:

```angular2html
class FakeRepository(repository.AbstractRepository):
    def __init__(self, batches):
        self._batches = set(batches)

    def add(self, batch):
        self._batches.add(batch)

    def get(self, reference):
        return next(b for b in self._batches if b.reference == reference)

    def list(self):
        return list(self._batches)

class FakeUnitOfWork(unit_of_work.AbstractUnitOfWork):
    def __init__(self):
        self.batches = FakeRepository([])
        self.committed = False

    def commit(self):
        self.committed = True

    def rollback(self):
        pass
```

**FakeUnitOfWork** and **FakeRepository** are tightly coupled, just like the real UnittofWork and Repository classes. That’s fine because we recognize that the objects
are collaborators.

Here is some tests in service layer:

```angular2html
def test_add_batch():
    uow = FakeUnitOfWork()
    services.add_batch("b1", "CRUNCHY-ARMCHAIR", 100, None, uow)
    assert uow.batches.get("b1") is not None
    assert uow.committed


def test_allocate_returns_allocation():
    uow = FakeUnitOfWork()
    services.add_batch("batch1", "COMPLICATED-LAMP", 100, None, uow)
    result = services.allocate("o1", "COMPLICATED-LAMP", 10, uow)
    assert result == "batch1"
```

In our tests, we can instantiate a UoW and pass it to our service layer, rather than
passing a repository and a session. This is considerably less cumbersome.

- **Using the UoW in the Service Layer**

Here’s what our new service layer looks like:

```angular2html
def is_valid_sku(sku, batches):
    return sku in {b.sku for b in batches}


def add_batch(
    ref: str, sku: str, qty: int, eta: Optional[date],
    uow: unit_of_work.AbstractUnitOfWork,
):
    with uow:
        uow.batches.add(model.Batch(ref, sku, qty, eta))
        uow.commit()


def allocate(
    orderid: str, sku: str, qty: int,
    uow: unit_of_work.AbstractUnitOfWork,
) -> str:
    line = OrderLine(orderid, sku, qty)
    with uow:
        batches = uow.batches.list()
        if not is_valid_sku(line.sku, batches):
            raise InvalidSku(f"Invalid sku {line.sku}")
        batchref = model.allocate(line, batches)
        uow.commit()
    return batchref
```

Our service layer now has only the one dependency, once again on an **abstract UoW**.

- **Examples: Using UoW to Group Multiple Operations into an Atomic Unit**

Here are a few examples showing the Unit of Work pattern in use. You can see how it
leads to simple reasoning about what blocks of code happen together.

- **Example 1: Reallocate**

Suppose we want to be able to deallocate and then reallocate orders:

```angular2html
def reallocate(line: OrderLine, uow: AbstractUnitOfWork) -> str:
   with uow:
     batch = uow.batches.get(sku=line.sku)
     if batch is None:
       raise InvalidSku(f'Invalid sku {line.sku}')
     batch.deallocate(line)
     allocate(line)
     uow.commit()
```
If **deallocate()** fails, we don’t want to call **allocate()**, obviously. And, If **allocate()** fails, we probably don’t want to actually commit the **deallocate()** either.

- **Example 2: Change Batch Quantity**

Our shipping company gives us a call to say that one of the container doors opened,
and half our sofas have fallen into the Indian Ocean. Oops!

```angular2html
def change_batch_quantity(batchref: str, new_qty: int, uow: AbstractUnitOfWork):
   with uow:
     batch = uow.batches.get(reference=batchref)
     batch.change_purchased_quantity(new_qty)
     while batch.available_quantity < 0:
       line = batch.deallocate_one()
     uow.commit()
```
Here we may need to deallocate any number of lines. If we get a failure at any
stage, we probably want to commit none of the changes.

## Chapter7: Aggregates and Consistency Boundaries
[Source Code](https://github.com/cosmicpython/code/tree/chapter_07_aggregate)

In this chapter, we’d like to revisit our domain model to talk about invariants and
constraints, and see how our domain objects can maintain their own internal consistency, both conceptually and in persistent storage.

Figure below shows a preview of where we’re headed: we’ll introduce a new model object
called **Product** to wrap multiple batches, and we’ll make the old **allocate()** domain
service available as a method on **Product** instead.

![](images/product_aggregate.png)

- **Invariants, Constraints, and Consistency**

constraint is a rule that restricts the possible states our model can get into, while an **invariant** is defined a little more
precisely as a condition that is always true. If we were writing a hotel-booking system, we might have the constraint that double
bookings are not allowed. This supports the invariant that a room cannot have more
than one booking for the same night. Let’s look at a couple of concrete examples from our business requirements; we’ll start
with this one: **An order line can be allocated to only one batch at a time.**

In a single-threaded, single-user application, it’s relatively easy for us to maintain this
invariant. We can just allocate stock one line at a time, and raise an error if there’s no
stock available. This gets much harder when we introduce the idea of concurrency. Suddenly we
might be allocating stock for multiple order lines simultaneously. We usually solve this problem by applying **locks** to our database tables. This prevents
two operations from happening simultaneously on the same row or same table.

As we start to think about scaling up our app, we realize that our model of allocating
lines against all available batches may not scale. If we process tens of thousands of
orders per hour, and hundreds of thousands of order lines, we can’t hold a lock over
the whole **batches** table for every single one we’ll get deadlocks or performance
problems at the very least.

- **What Is an Aggregate?**

The **Aggregate** pattern is a design pattern from the DDD community that helps us to
resolve this tension. An **aggregate** is just a domain object that contains other domain
objects and lets us treat the whole collection as a single unit. it’s a good idea to nominate some entities to be the single entrypoint for modi‐
fying their related objects. It makes the system conceptually simpler and easy to
reason about if you nominate some objects to be in charge of consistency for the others.

For example, if we’re building a shopping site, the **Cart** might make a good aggregate:
it’s a collection of items that we can treat as a single unit. Importantly, we want to load
the entire basket as a single blob from our data store. We don’t want two requests to
modify the basket at the same time, or we run the risk of weird concurrency errors.
Instead, we want each change to the basket to run in a single database transaction. We don’t want to modify multiple baskets in a transaction, because there’s no use case
for changing the baskets of several customers at the same time. Each basket is a single
consistency boundary responsible for maintaining its own invariants.So, An **AGGREGATE** is a cluster of associated objects that we treat as a unit for the purpose of data changes.
our aggregate has a root entity (the Cart) that encapsulates access to items.
Each item has its own identity, but other parts of the system will always refer to the Cart only as an indivisible whole.

- **Choosing an Aggregate**

What aggregate should we use for our system? The choice is somewhat arbitrary, but
it’s important. The aggregate will be the boundary where we make sure every operation ends in a consistent state. This helps us to reason about our software and prevent
weird race issues.

The following example is constructed by **ChatGPT**.

- **Simple Example: Shopping Cart**

Consider a simple e-commerce system with a **ShoppingCart** aggregate. Here's how the aggregate ensures consistency:

- **Entities and Value Objects:**

1- **ShoppingCart (Aggregate Root):** Contains items, a total cost, and methods to add or remove items.

2- **CartItem (Entity within Aggregate):** Represents an individual item in the cart, including product details and quantity.

3- **ProductID (Value Object):** A unique identifier for a product.

- **Business Rule (Invariant)**

The total cost of the cart must always equal the sum of the costs of all items in the cart.

```angular2html
from pydantic import BaseModel
from typing import List, Dict

class ProductID(BaseModel):
    id: str

class CartItem(BaseModel):
    product_id: ProductID
    quantity: int
    price_per_unit: float
    
    def total_price(self):
        return self.quantity * self.price_per_unit

class ShoppingCart(BaseModel):
    items: Dict[str, CartItem]  # key is ProductID.id
    
    def add_item(self, product_id: ProductID, quantity: int, price_per_unit: float):
        if product_id.id in self.items:
            self.items[product_id.id].quantity += quantity
        else:
            self.items[product_id.id] = CartItem(product_id=product_id, quantity=quantity, price_per_unit=price_per_unit)
        self.check_invariants()
    
    def remove_item(self, product_id: ProductID, quantity: int):
        if product_id.id in self.items:
            if quantity >= self.items[product_id.id].quantity:
                del self.items[product_id.id]
            else:
                self.items[product_id.id].quantity -= quantity
        self.check_invariants()
    
    def total_cost(self):
        return sum(item.total_price() for item in self.items.values())
    
    def check_invariants(self):
        if self.total_cost() < 0:
            raise ValueError("Total cost cannot be negative.")

```

In this example:

1- The **ShoppingCart** class (aggregate root) manages all modifications to the cart's contents.

2- It ensures the invariant (total cost is correct) is maintained after any changes such as adding or removing items.

3- All interactions with **CartItem** objects go through the **ShoppingCart**, which controls and checks the consistency of its internal state.
This model allows the application to maintain consistent rules about how products are added or removed, ensuring data integrity and business rule enforcement within the shopping cart system.

(According to the previous rule) only Aggregate Root can be directly retrieved from the database by Query. In other words, a Repository is defined for each Aggregate Root, and other objects cannot be retrieved directly and must be retrieved by the corresponding Aggregate Root. Also, direct access to save or edit an entity inside an Aggregate is meaningless, and database transactions are meaningful at the level of an Aggregate.

- **Aggregates, Bounded Contexts, and Microservices**

One of the most important contributions from Evans and the DDD community is the concept of **bounded contexts**.
In essence, this was a reaction against attempts to capture entire businesses into a single model. The word customer means different things to people in sales, customer ser‐
vice, logistics, support, and so on. Attributes needed in one context are irrelevant in
another; more perniciously, concepts with the same name can have entirely different
meanings in different contexts. Rather than trying to build a single model (or class, or
database) to capture all the use cases, it’s better to have several models, draw boundaries around each context, and handle the translation between different contexts
explicitly. This concept translates very well to the world of microservices, where each microservice is free to have its own concept of “customer” and its own rules for translating that
to and from other microservices it integrates with. Whether or not you have a microservices architecture, a key consideration in choos‐
ing your aggregates is also choosing the bounded context that they will operate in. By
restricting the context, you can keep your number of aggregates low and their size
manageable.

- **One Aggregate = One Repository**

Once you define certain entities to be aggregates, we need to apply the rule that they
are the only entities that are publicly accessible to the outside world. In other words,
the only repositories we are allowed should be repositories that return aggregates.

In our case, we’ll switch from **BatchRepository** to **ProductRepository**:

```angular2html
class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, product: model.Product):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, sku) -> model.Product:
        raise NotImplementedError


class AbstractUnitOfWork(abc.ABC):
    products: repository.AbstractRepository

    def __enter__(self) -> AbstractUnitOfWork:
        return self

    def __exit__(self, *args):
        self.rollback()

    @abc.abstractmethod
    def commit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def rollback(self):
        raise NotImplementedError
```

Service layer

```angular2html
def add_batch(
    ref: str, sku: str, qty: int, eta: Optional[date],
    uow: unit_of_work.AbstractUnitOfWork,
):
    with uow:
        product = uow.products.get(sku=sku)
        if product is None:
            product = model.Product(sku, batches=[])
            uow.products.add(product)
        product.batches.append(model.Batch(ref, sku, qty, eta))
        uow.commit()


def allocate(
    orderid: str, sku: str, qty: int,
    uow: unit_of_work.AbstractUnitOfWork,
) -> str:
    line = OrderLine(orderid, sku, qty)
    with uow:
        product = uow.products.get(sku=line.sku)
        if product is None:
            raise InvalidSku(f"Invalid sku {line.sku}")
        batchref = product.allocate(line)
        uow.commit()
    return batchref
```

- **What About Performance?**

We’ve mentioned a few times that we’re modeling with aggregates because we want to
have high-performance software, but here we are loading all the batches when we
only need one. You might expect that to be inefficient, but there are a few reasons
why we’re comfortable here.

First, we’re purposefully modeling our data so that we can make a single query to the
database to read, and a single update to persist our changes. This tends to perform
much better than systems that issue lots of ad hoc queries. In systems that don’t
model this way, we often find that transactions slowly get longer and more complex
as the software evolves.

Second, we expect to have only 20 or so batches of each product at a time. Once a batch
is used up, we can discount it from our calculations. This means that the amount of
data we’re fetching shouldn’t get out of control over time.

The Aggregate pattern is designed to help manage
some technical constraints around consistency and performance. There isn’t one correct aggregate, and we should feel comfortable changing our minds if we find our
boundaries are causing performance woes.

- **Optimistic Concurrency with Version Numbers**

will be added soon ...

- **Implementation Options for Version Numbers**

will be added soon ...

- **Testing for Our Data Integrity Rules**

will be added soon ...

- **Pessimistic Concurrency Control Example: SELECT FOR UPDATE**

will be added soon ...

- **Wrap-Up**

we explicitly model an object as being the main
entrypoint to some subset of our model, and as being in charge of enforcing the
invariants and business rules that apply across all of those objects. Choosing the right aggregate is key, and it’s a decision you may revisit over time.

1- Aggregates are your entrypoints into the domain model

2- Aggregates are in charge of a consistency boundary


- **Part I Recap**

![](images/component_diagram_end_part_one.png)

We’ve decoupled the infrastructural parts of our system, like the database and API
handlers, so that we can plug them into the outside of our application. This helps us
to keep our codebase well organized and stops us from building a big ball of mud.

By applying the dependency inversion principle, and by using ports-and-adapters inspired patterns like Repository and Unit of Work, we’ve made it possible to do TDD
in both high gear and low gear and to maintain a healthy test pyramid. We can test
our system edge to edge, and the need for integration and end-to-end tests is kept to a
minimum.

Lastly, we’ve talked about the idea of consistency boundaries. We don’t want to lock
our entire system whenever we make a change, so we have to choose which parts are
consistent with one another.

For a small system, this is everything you need to go and play with the ideas of
domain-driven design. You now have the tools to build database-agnostic domain
models that represent the shared language of your business experts. Hurrah!

At the risk of laboring the point—we’ve been at pains to point out
that each pattern comes at a cost. Each layer of indirection has a
price in terms of complexity and duplication in our code and will
be confusing to programmers who’ve never seen these patterns
before. If your app is essentially a simple CRUD wrapper around a
database and isn’t likely to be anything more than that in the foreseeable future, you don’t need these patterns. Go ahead and use
Django, and save yourself a lot of bother.


# Event-Driven Architecture