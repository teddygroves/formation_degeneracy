.phony: clean-paper clean-all clean-stan-output paper

BIBLIOGRAPHY = bibliography.bib
PANDOCFLAGS =                         \
  --from=org                          \
  --highlight-style=pygments          \
  --pdf-engine=xelatex                \
  --bibliography=$(BIBLIOGRAPHY)
RESULTS_TOY = results/toy_case_study
RESULTS_BIG = results/big_case_study
SAMPLES_TOY = $(shell find $(RESULTS_TOY)/samples -name "*.csv")
SAMPLES_BIG = $(shell find $(RESULTS_BIG)/samples -name "*.csv")
LOGS_TOY = $(shell find $(RESULTS_TOY)/samples -name "*.txt")
LOGS_BIG = $(shell find $(RESULTS_BIG)/samples -name "*.txt")
LAST_UPDATED_TOY = $(RESULTS_TOY)/samples/last_updated.txt
LAST_UPDATED_BIG = $(RESULTS_BIG)/samples/last_updated.txt
REPARAM_FILE = $(RESULTS_BIG)/equilibrator_reparam.json
SCRIPT_TOY = run_toy_simulation_study.py
SCRIPT_BIG = run_big_simulation_study.py
MODEL_NAIVE = model_naive
MODEL_SMART = model_smart

report.pdf: report.org $(BIBLIOGRAPHY)
	pandoc $< -o $@ $(PANDOCFLAGS)

paper: report.pdf

$(LAST_UPDATED_TOY): $(SCRIPT_TOY) $(MODEL_NAIVE).stan $(MODEL_SMART).stan
	python3 $<

$(LAST_UPDATED_BIG): $(SCRIPT_BIG) $(MODEL_NAIVE).stan $(MODEL_SMART).stan
	python3 $<

samples-toy: $(LAST_UPDATED_TOY)

samples-big: $(LAST_UPDATED_BIG)

samples: samples_toy, samples_big

clean-paper: 
	$(RM) *.pdf

clean-stan:
	$(RM) $(MODEL_NAIVE) $(MODEL_NAIVE).hpp $(MODEL_SMART) $(MODEL_SMART).hpp

clean-samples-toy: 
	$(RM) $(SAMPLES_TOY) $(LOGS_TOY) $(LAST_UPDATED_TOY)

clean-samples-big: 
	$(RM) $(SAMPLES_BIG) $(LOGS_BIG) $(LAST_UPDATED_BIG)

clean-big-reparam:
	$(RM) $(REPARAM_FILE)

clean-samples: clean-samples-toy clean-samples-big

clean-all: clean-paper clean-samples clean-stan clean-big-reparam
