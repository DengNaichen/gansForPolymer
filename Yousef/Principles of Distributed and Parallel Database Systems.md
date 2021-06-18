# Why distribution
In order to achieve 4 goals:

1. Availability: To make sure the system is 100% available for users anytime since you have multiple systems and components.

2. Scalability: Easy to expend number of users and amount of data.

3. Reliability/Fault Tolerance: 

4. Transparency: isolates all the system details from users. The users don't see that happens in the system. They just ask query like centralized systems.

What is distributed database system?
1. a collection of multiple, **logically interrelated(互相关联的)** database distributed over a **computer network**.
2. The software that manage the DDB and provides an access mechanism that makes this distribution transparent to the users.

DDBMS promises:
1. **transparent management** of distributed, fragmented(零散的), and replicated (复制的)data.
2. improved **reliability/availability** through distributed transactions.
3. **improved performance**.
4. Easier and more economical system **expansion**.

## Data Fragmentation and Replication Models
why we need fragmentation?
If we have a centralized database, then we want to make it distributed. easy to store and reduce the storage of hard disk.

### good fragmentation
1. completeness: 所有在R中可以找到的元素，都可以在R分成的Ri中找到。
2. reconstruction: 被分割的方式应该遵循某些规律。
3. disjointness: 被分出的部分应该不能用重复

### Horizontal fragmentation
![](https://i.imgur.com/JUcUmUv.png)
有不一样的算法可以来分割，不想具体举例子了。
### Vertical fragmentation
Attribute affinities（相似度）
1. a measure that indicates how closely related the attributes are 
2. this is obtained from more primitive(原始的) usage data.

Attribute usage values
![](https://i.imgur.com/Y9eNioX.png)

attribute values can be represented by a matrix.

![](https://i.imgur.com/lqChhaZ.png)

so we can use the attribute affinity to do something, for example:

The attribute affinity measure between two attributes $A_{i}$ and $A_{j}$ of a relation R with respect to the set of applications Q is defined as:

![](https://i.imgur.com/Le2CFai.png)

Then we can get an attribute affinity matrix, where the horizontal and vertical are both attributes, and the values in the matrix is the similarity. we can put high affinity attributes in one fragment together.

*Question: need more example to know how to calculate the affinity table.*

### Replication
why replication
1. increased availability: for example, one site goes down, the other one still running.
2. Faster query evaluation: too many query request needs to be answer, then we can run them on different replicas.
3. Potentially improved performance: increase the proximity of data to its points of use, then requires some support for fragmentation and replication. (put the data in the closer server)
4. Challenge: Updating

Allocation Alternatives:
- Non-replicated: shared, partitioned: each fragment resides at only one site.
- Replicated: 
    - fully replicated: each fragment at each site.
    - partially replicated: each fragment at some of the sites

if read-only/update >> 1, then replication is advantageous, we will also need to address lots of issues. (遇到再说吧)

## Advanced Distributed Database Systems
### query processing in a distributed database
![](https://i.imgur.com/he22H37.png)
1. high level user query: query language that is used: SQL
2. Query execution methodology: the steps that one goes through in executing high-level(declarative) user queries.
3. Query optimization: How do we determine the 'best" execution plan.

especially for the distributed db system, we need count the transaction cost as well, 在移动计算里学过其实传输消耗的time and energy很多时候会比本地计算还要多得多。

for example we have a query:
![](https://i.imgur.com/cjlCEXt.png)
if we run it on centralized database, we may have two different strategy:
![](https://i.imgur.com/Faodeuy.png)

and if we consider a distributed system, like following:
![](https://i.imgur.com/9HKUGdh.png)
there are lots of strategy as well, for example:
![](https://i.imgur.com/8AUCLBr.png)
and 
![](https://i.imgur.com/DqNFJ35.png)
different size of data will lead different cost, therefore developer need to build a cost model to achieve a better performance. usually we need to consider several factor:
1. solution space: the set of equivalent algebra expressions
2. cost function: 
    - i/o cost + cpu cost
    - different weights in different distributed environments
    - maximize throughput(吞吐量)
3. search algorithm:
    - move inside the solution space
    - exhaustive search





## Parallel Database Systems
17mins
## Quiz
