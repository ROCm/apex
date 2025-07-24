PYTHON = python3
PIP = $(PYTHON) -m pip

clean: # This will remove ALL build folders.
	@rm -r build/
	@rm -r dist/
	@rm -r *.egg-info
aiter:
	$(PIP) uninstall -y aiter
	cd third_party/aiter && $(PIP) install . --no-build-isolation --no-deps
	
