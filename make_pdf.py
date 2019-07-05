#!/usr/bin/env python3

import subprocess
import glob

title = '深度学习500问'
directories = sorted(glob.glob('./ch*'))
md_files = sorted(glob.glob('./ch*/第*.md'))
resource_path = ':'.join(directories)
input_files = ' '.join(['"' + x + '"' for x in md_files])

subprocess.call('pandoc --pdf-engine=xelatex --metadata "title:' + title + '" --resource-path "' + resource_path + '" -o deeplearning.pdf -i ' + input_files, shell=True)
