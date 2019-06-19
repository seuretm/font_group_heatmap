## Grad-CAM implementation in Pytorch ##

Please note that the vaste majority of heatmap.py comes from the
following repository by Jacob Gildenblat:
https://github.com/jacobgil/pytorch-grad-cam

See the Grad-CAM paper:
https://arxiv.org/pdf/1610.02391v1.pdf
The paper authors torch implementation:
https://github.com/ramprs/grad-cam
Jacob Gildenblat's Keras implementation:
https://github.com/jacobgil/keras-grad-cam

The code has been adapted to process input images too large to fit in
the memory of currently existing GPUs by processing the data patch-wise.
Note that depending on the CNN architecture which is used, there might
be some border effects. This is the case, for example, for residual
networks. Dense networks seem not to have this issue.

A second modification is the normalization of all heatmaps together to
allow them to be compared.

Finally, instead of using directlythe CNN, it uses the following
classifier:
https://github.com/seuretm/ocrd_typegroups_classifier

Usage:
`python grad-cam.py --image-path <path_to_image>`

Note that:
- an "output" folder is needed.
- it should work with PyTorch 1.0
