#!/usr/bin/python
import errno
import os
import subprocess
import shutil
import sys


def MakeDirsExistOk(directory_path):
  try:
    os.makedirs(directory_path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise


def Call(command):
  print('Calling: ' + command)
  proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  stdout, stderr = proc.communicate()
  if proc.returncode != 0:
    print('Call failed with error code ' + str(proc.returncode))
    sys.exit(1)


def CallWithTiming(command):
  # This measures the time in seconds.
  command_with_timing = '/usr/bin/time -f \"%e\" ' + command
  
  print('Calling: ' + command_with_timing)
  proc = subprocess.Popen(command_with_timing, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
  stdout, stderr = proc.communicate()
  if proc.returncode != 0:
    print('Call failed with error code ' + str(proc.returncode))
    sys.exit(1)
  
  return float(stderr.decode("utf-8"))


# This script must be run separately on the high_res_multi_view and
# low_res_many_view folders. It is intended to be used with the undistorted
# versions of the images; otherwise, change how "input_path" is determined
# below.
if __name__ == '__main__':
  if len(sys.argv) != 4:
    print('Usage: python eval-mvs-colmap.py <colmap_binary_path> <datasets_path> <output_path>')
    sys.exit(1)
  
  colmap_binary_path = sys.argv[1]
  datasets_path = sys.argv[2]
  output_path = sys.argv[3]
  
  MakeDirsExistOk(output_path)
  
  # Loop over all directories in datasets_path.
  for dir_name in os.listdir(datasets_path):
    dir_path = os.path.join(datasets_path, dir_name)
    if not os.path.isdir(dir_path):
      continue
    
    # Skip if output already present.
    output_file_path = os.path.join(output_path, dir_name + '.ply')
    timing_file_path = os.path.join(output_path, dir_name + '.txt')
    if os.path.isfile(output_file_path) and os.path.isfile(timing_file_path):
      print('Skipping since output already present: ' + dir_name)
      continue
    
    print('Processing: ' + dir_name)
    
    # Call COLMAP.
    image_path = os.path.join(dir_path, 'images')
    input_path = os.path.join(dir_path, 'dslr_calibration_undistorted')
    workspace_path = os.path.join(output_path, 'temp')
    
    call = (os.path.join(colmap_binary_path, 'image_undistorter') +
            ' --image_path ' + image_path +
            ' --input_path ' + input_path +
            ' --output_path ' + workspace_path)
    Call(call)
    
    call = (os.path.join(colmap_binary_path, 'dense_stereo') +
            ' --workspace_path ' + workspace_path +
            ' --DenseStereo.max_image_size 2000')
    duration = CallWithTiming(call)
    
    call = (os.path.join(colmap_binary_path, 'dense_fuser') +
            ' --output_path ' + output_file_path +
            ' --workspace_path ' + workspace_path)
    duration += CallWithTiming(call)
    
    # Write timing file.
    with open(timing_file_path, 'w') as timing_file:
      timing_file.write('runtime ' + str(duration))
    
    # Delete workspace path.
    shutil.rmtree(workspace_path)
