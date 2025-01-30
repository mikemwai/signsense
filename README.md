# Signsense

This is a machine learning model designed for the recognition of Kenyan Sign Language. This project aims to bridge the communication gap by providing an efficient and accessible tool for understanding and interpreting Kenyan Sign Language.

![Flask Version](https://img.shields.io/badge/Flask-3.1%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Structure

```
signsense/
├── Dataset/
│   ├── Dataset/
│   ├── Dataset.md
│   └── Dataset_Visual.png
├── Logs/
├── database/
│   ├── DATABASE.md
│   ├── config.py
│   ├── create_db.py
│   └── models.py
├── model/
│   ├── Model.h5
│   ├── Model.md
│   ├── comparative_analysis_output.png
│   ├── confusionmatrix_output.png
│   ├── metrics.png
│   └── signsense_mediapipe_lstm.ipynb
├── routes/
│   └── routes.py
├── static/
│   ├── css/
│   ├── images/
│   ├── js/
│   └── logo.png
├── templates/
│   ├── layouts/
│   └── pages/
├── utilities/
│   ├── extensions.py
│   └── utils.py
├── .gitignore
├──  LICENSE.txt
├── app.py
└── requirements.txt
```

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributions](#contributions)
4. [Issues](#issues)
5. [License](#license)
6. [Database](./database/DATABASE.md)
7. [Model](./model/Model.md)
8. [Dataset](./Dataset/Dataset.md)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/mikemwai/signsense.git
    ```
2. Navigate to the project directory:
    ```sh
    cd signsense
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

    - Picks up all the packages from the project and copies to the `requirements.txt` file:
      
        ```sh
        pip freeze >> requirements.txt
        ```

## Usage

- Run the following command to start the application on your local machine:

  - On Windows:

    ```sh
        set FLASK_APP=app.py
        flask run --host=0.0.0.0
    ```

  - On Unix/Linux/Mac:

    ```sh
        export FLASK_APP=app.py
        flask run
    ```

- This will start a development server on http://127.0.0.1:5000/ where you can access the application.

## Contributions

If you'd like to contribute to this project:

- Please fork the repository.
- Create a new branch for your changes.
- Submit a [pull request](https://github.com/mikemwai/signsense/pulls).

Contributions, bug reports, and feature requests are welcome!

## Issues

If you have any issues with the project, feel free to open up an [issue](https://github.com/mikemwai/signsense/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
