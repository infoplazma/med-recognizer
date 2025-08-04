---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	agent1(agent1)
	agent2(agent2)
	combiner(combiner)
	completion(completion)
	__end__([<p>__end__</p>]):::last
	__start__ --> agent1;
	__start__ --> agent2;
	agent1 -.-> combiner;
	agent2 -.-> combiner;
	combiner -.-> __end__;
	combiner -.-> completion;
	completion -.-> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
