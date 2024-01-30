@echo off

mkdir "downloads"
cd "downloads"
DEL  "*.tsv"
cd ".."
REM UNKNOWN: {"type":"For","name":{"text":"var","type":"Name"},"wordlist":[{"text":"name.basics","type":"Word"},{"text":"title.akas","type":"Word"},{"text":"title.basics","type":"Word"},{"text":"title.crew","type":"Word"},{"text":"title.episode","type":"Word"},{"text":"title.principals","type":"Word"},{"text":"title.ratings","type":"Word"}],"do":{"type":"CompoundList","commands":[{"type":"Pipeline","commands":[{"type":"Command","name":{"text":"curl","type":"Word"},"suffix":[{"text":"https://datasets.imdbws.com/$var.tsv.gz","expansion":[{"loc":{"start":28,"end":31},"parameter":"var","type":"ParameterExpansion"}],"type":"Word"}]},{"type":"Command","name":{"text":"zcat","type":"Word"},"suffix":[{"type":"Redirect","op":{"text":">","type":"great"},"file":{"text":"./downloads/$var.tsv","expansion":[{"loc":{"start":12,"end":15},"parameter":"var","type":"ParameterExpansion"}],"type":"Word"}}]}]}]}}
