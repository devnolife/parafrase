from setuptools import setup, find_packages

setup(
    name='parafrase',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # dependencies utama, bisa juga otomatis dari requirements.txt
    ],
    include_package_data=True,
    description='Sistem Parafrase Bahasa Indonesia (Hybrid + IndoT5)',
    author='Nama Anda',
    author_email='email@domain.com',
    url='https://github.com/username/parafrase',
) 
