from setuptools import setup, find_packages


setup(
     name='chainermnx',
     version='0.1',
     author="Albert Kahira",
     author_email="ankahira@gmail.com",
     description="Extended ChainerMN",
     long_description_content_type="Extended ChainerMN to facilate different forms of model parallelism ",
     url="https://github.com/ankahira/chainermnx",
     packages=find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     install_requires=['chainer', 'numpy']
 )
setup()
