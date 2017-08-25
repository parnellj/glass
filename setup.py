try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'name': 'Glass',
	'version': '0.1',
	'url': 'https://github.com/parnellj/glass',
	'download_url': 'https://github.com/parnellj/glass',
	'author': 'Justin Parnell',
	'author_email': 'parnell.justin@gmail.com',
	'maintainer': 'Justin Parnell',
	'maintainer_email': 'parnell.justin@gmail.com',
	'classifiers': [],
	'license': 'GNU GPL v3.0',
	'description': 'Analyzes prosody and grammar within and between Emily Dickinsons poems.',
	'long_description': 'Analyzes prosody and grammar within and between Emily Dickinsons poems.',
	'keywords': '',
	'install_requires': ['nose'],
	'packages': ['glass'],
	'scripts': []
}
	
setup(**config)
