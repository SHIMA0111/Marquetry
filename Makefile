cleanup:
	@rm -rf ./build ./dist ./marquetry.egg-info

build: cleanup
	@rm -rf ./artifacts
	@python -m build
	@mkdir artifacts
	@cp ./dist/* ./artifacts/
	@make cleanup

testupload:
	@twine upload --repository testpypi artifacts/*

testrelease: build testupload

upload:
	@echo "Product type upload can't be executed by make command for safety."
	@echo "You want to upload the library, please execute 'twine upload --repository pypi artifacts/*' after 'make build'."
