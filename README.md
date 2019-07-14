# sma-crossover-strategy
Project to analyze finance of various brands.

- You may use smaenv virtual environment to use the required packages:
	./smaenv/bin/activate (Windows)
	source smaenv/bin/activate (Linux/ Mac)
- Download git from https://git-scm.com/download/win to use git from git bash/ CMD.


1. Fork the repo. (not required now)
2. Clone the repo into your local machine. Browse into your projects directory then:
	git clone https://github.com/username/sma-crossover-strategy.git
	(replace the username with your username)
3. Start working on your local machine (working directory).
4. To check status of the git:
	git status
5. After editing, to add files to the staging area:
	git add -A (adds/ updates all the files, subfolders, deleted files, modified files, new files (and parent directory too, if under git) to the staging area)
	git add . (adds/ updates all the files, subfolders, deleted files, modified files to the staging area. It doesn't add parent directory.)
	git add -u (same as -A except it does not add new files)
	git add * (not a git thing. * is an OS thing. It doesn't add deleted files and parent folders too)
6. Staging area -> local repo
	git commit -m 'message here'
7. Check global config file:
	git config --list
8. If user.name and user.email is not there:
	git config --global user.name 'user-name here'
	git config --global user.email 'user-email here'
9. Check remote:
	git remote get-url origin (it should be there, I guess)
10. If not there:
	git remote add origin 'https://github.com/username/sma-crossover-strategy.git' (your username)
11. Before pushing your local repo to remote repo, pull from remote to have updated things:
	git pull origin master
	git push origin master
12. If push doesn't work,
	git push -u origin master
