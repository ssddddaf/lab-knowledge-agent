import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from lab_agent.models import RunInput
from lab_agent.store import Store


@contextmanager
def case_folder():
    folder=Path("tests") / (".test_"+uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()


class StoreTests(unittest.TestCase):
    def test_document_dedup_and_lineage(self):
        with case_folder() as folder:
            store=Store(folder/"test.db")
            store.add_document("sha","one.pdf",str(Path(folder)/"one.pdf"))
            store.add_document("sha","duplicate.pdf",str(Path(folder)/"duplicate.pdf"))
            self.assertEqual(len(store.documents()),1)
            base=dict(input_version="v1",method="tomography",parameters={"grid_m":50},status="completed",timestamp="2026-03-01T00:00:00",artifact="velocity-v1")
            store.add_run("demo",RunInput(run_id="r1",**base))
            with self.assertRaises(ValueError):
                store.add_run("demo",RunInput(run_id="r2",parent_run_id="missing",**base))
            store.add_run("demo",RunInput(run_id="r2",parent_run_id="r1",**base))
            self.assertEqual(store.lineage("demo")[1]["parent_run_id"],"r1")
            self.assertTrue(store.lineage("demo")[0]["simulated"])

    def test_block_index_and_job(self):
        with case_folder() as folder:
            store=Store(folder/"test.db")
            store.add_document("sha","paper.pdf","paper.pdf")
            store.add_page("sha",1,"page.png","velocity 3 km/s")
            store.add_blocks([{"id":"sha:1:a","document_id":"sha","page_number":1,"kind":"text","text":"velocity 3 km/s","bbox":[0,0,10,10]}])
            store.set_document_status("sha","ready",pages=1)
            self.assertEqual(store.blocks()[0]["id"],"sha:1:a")
            store.set_job("job","sha","queued")
            store.set_job("job","sha","done")
            self.assertEqual(store.job("job")["status"],"done")
            store.set_query("q","running")
            store.add_query_event("q","plan")
            store.add_query_event("q","retrieve")
            self.assertEqual([e["node"] for e in store.query_events("q")],["plan","retrieve"])
            cursor=store.query_events("q")[0]["seq"]
            self.assertEqual(len(store.query_events("q",cursor)),1)


if __name__=="__main__":
    unittest.main()
