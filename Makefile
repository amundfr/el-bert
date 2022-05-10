BASE_DIR = /nfs/students/amund-faller-raheim/ma_thesis_el_bert

help:
	@echo ""
	@echo "Available make commands are:"
	@echo " help          : prints this message"
	@echo " build         : builds image using Wharfer"
	@echo " run           : runs container using Wharfer."
	@echo "                    You will be greeted by a new Makefile inside."
	@echo " run-full      : runs container using Wharfer, and runs "
	@echo "                    setup, unittest, train and evaluation"
	@echo " run-unittest  : runs container using Wharfer, and runs unittests"
	@echo ""

build:
	wharfer build -t el_bert .

run:
	wharfer run -v $(BASE_DIR)/models:/models -v $(BASE_DIR)/data:/data -v $(BASE_DIR)/ex_data:/ex_data -it el_bert

run-full:
	wharfer run -v $(BASE_DIR)/models:/models -v $(BASE_DIR)/data:/data -v $(BASE_DIR)/ex_data:/ex_data -it el_bert /bin/sh -c "make full"

run-unittest:
	wharfer run -v $(BASE_DIR)/models:/models -v $(BASE_DIR)/data:/data -v $(BASE_DIR)/ex_data:/ex_data -it el_bert /bin/sh -c "make unittest"
