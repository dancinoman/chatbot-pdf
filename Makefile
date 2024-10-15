
first_run:
	pyenv virtualenv 3.10.6 le-package

start_session:
	pyenv global le-package && pip install -r requirements.txt
