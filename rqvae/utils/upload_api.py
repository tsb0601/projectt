import asyncio
import sys
async def upload_file(args: tuple):
    # Construct the command with the arguments
    command = [sys.executable, './rqvae/utils/upload.py'] + list(args)
    print(f'[!] ASYNC COMMAND: {command}')
    # Create the subprocess
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Capture the output
    stdout, stderr = await process.communicate()

    if process.returncode == 0:
        print(f'[!] ASYNC SUCCEED: {stdout.decode()}')
    else:
        print(f'[!] ASYNC FAIL: {stderr.decode()}')

async def upload_main_process(args: tuple):
    await upload_file(args)

def asyncio_GCS_op(instr:str, possible_file_or_dir_name:str = None, possible_destination:str = None):
    args = (instr, possible_file_or_dir_name, possible_destination)
    args = [arg for arg in args if arg is not None]
    asyncio.run(upload_main_process(args))
