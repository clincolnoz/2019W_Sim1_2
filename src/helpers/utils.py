# -*- coding: utf-8 -*-
import asyncio
from sys import stdout


def print_progress(percent: float):
    """
    Print progress.
    :param percent:
    """
    stdout.write("\r")
    stdout.write("    {:.0f}%".format(percent * 100))
    stdout.flush()


async def gather_dict(tasks: dict):
    """
    Gathers dictionary results.
    :param tasks:
    :return:
    """
    async def mark(key, coro):
        return key, await coro

    return {
        key: result
        for key, result in await asyncio.gather(
            *(mark(key, coro) for key, coro in tasks.items())
        )
    }
