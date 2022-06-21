<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://user-images.githubusercontent.com/59787386/174702243-5b2aaf4c-2e85-4dfe-a232-0ee37f964533.png" alt="Logo" width="850" height="250">
  </a>

<h3 align="center">Replicating Unsup3D</h3>

  <p align="center">
    Project for KAIST CS492(A): Machine Learning for 3D Data (22' Spring), by Yuseung Lee and Inhee Lee.
    <br /><br />
    <a href="https://github.com/phillipinseoul/unsup3d-rep/blob/main/report.pdf">Report</a>
    ·
    <a href="https://github.com/phillipinseoul/unsup3d-rep/blob/main/supplementary.pdf">Results</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
In this project, we present the reproduction results of [Unsup3D](https://arxiv.org/abs/1911.11130) from scratch and the challenges we faced during the implementation. The objective of Unsup3D is to learn the underlying 3D shape from single 2D images without any supervision. It decomposes the input into four visual components and reconstructs it through photo-geometric autoencoder. To obatain a geometric cue, Unsup3D applies symmetric assumption and introduces a confidence map. We implemented most parts of Unsup3D excluding the neural renderer and resolved CUDA-related issues with gradient clipping. Our implementation achieves close performance to the original paper and the author’s code trained on our environment.

### Built With

* Python
* PyTorch

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_


<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png


# Reproducing Unsup3D for CS492(A): Machine Learning for 3D Data (22' Spring)

By Yuseung Lee & Inhee Lee

## Environment Setting
We recommend you to use this model 

## How to run the code
You need to modify configs file first before train or test the model. We recommend you to use bfm_template.yml in configs/ablation/ as template.

We provide dataloader for BFM datasets and CelebA. 
Here is the link for both datasets. (This link will be closed after evaluation of CS492(A))

dataset gdrive : https://drive.google.com/drive/folders/1AzvmOGHv-4xLKotcAaP0pkeALRoMkGNp?usp=sharing
pretrained gdrive : https://drive.google.com/drive/folders/17101Jj5PcCqmb1ywRzmcriONTGOHZCDj?usp=sharing


codes to be runned
```bash
$ python run.py --configs configs/bfm_train_v0.yaml
$ tensorboard --logdir /logs/ --port 6001
```

* exmaples on BFM and CelebA
![image](https://user-images.githubusercontent.com/65122489/172181746-95db1bf6-a59f-41de-ace2-4067cad181a6.png)
* Pipdline of Unsup3D
![image](https://user-images.githubusercontent.com/65122489/172181610-a4b4ea31-a425-4751-b01f-ba0104d558cb.png)


## Reference
- Official Implementation
https://github.com/elliottwu/unsup3d
- Modified Neural Renderer
https://github.com/adambielski/neural_renderer
- Unsup3D[Wu et al.]
https://arxiv.org/abs/1911.11130


