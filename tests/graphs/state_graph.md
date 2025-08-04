---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	extractor(extractor)
	metadata_analyzer(metadata_analyzer)
	combiner(combiner)
	__end__([<p>__end__</p>]):::last
	__start__ --> extractor;
	__start__ --> metadata_analyzer;
	extractor --> combiner;
	metadata_analyzer --> combiner;
	combiner --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
