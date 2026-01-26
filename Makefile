PYTHON = python3
PIP = $(PYTHON) -m pip

clean: # This will remove ALL build folders.
	@test -d build/ && echo "Deleting build folder" || true
	@test -d build/ && rm -r build/ || true
	@test -d dist/ && echo "Deleting dist folder" || true
	@test -d dist/ && rm -r dist/ || true
	@test -d apex.egg-info/ && echo "Deleting apex.egg-info folder" || true
	@test -d apex.egg-info/ && rm -r apex.egg-info/ || true

	$(PYTHON) scripts/clean.py # remove the apex extensions installed at torch extensions folder 

aiter:
	$(PIP) uninstall -y aiter
	cd third_party/aiter && $(PIP) install . --no-build-isolation --no-deps
	
