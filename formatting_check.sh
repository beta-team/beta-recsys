flake8 . --count --select BLK --show-source --statistics
flake8 . --count --ignore=E402,W503,W504,E203,C901,I001,I003,I004,I100,I101,I201,D100 --max-complexity=10 --max-line-length=127 --statistics
python -m pytest --cov-report=xml --cov=./
isort . --check-only --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88