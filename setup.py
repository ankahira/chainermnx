from setuptools import setup, find_packages, Command
import os


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root: %s' % self.cwd
        os.system('rm -rf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


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
     install_requires=['chainer', 'numpy'],
     cmdclass={
        'clean': CleanCommand,
     }

 )

