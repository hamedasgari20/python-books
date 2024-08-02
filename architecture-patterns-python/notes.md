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

