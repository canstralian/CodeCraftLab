import unittest
from unittest.mock import patch


class TestStreamlitApp(unittest.TestCase):
    """Test suite for the Streamlit app components"""

    @patch("streamlit.title")
    @patch("streamlit.markdown")
    @patch("streamlit.subheader")
    @patch("streamlit.columns")
    @patch("streamlit.metric")
    @patch("utils.set_page_config")
    @patch("utils.display_sidebar")
    def test_app_initialization(
        self,
        mock_display_sidebar,
        mock_set_page_config,
        mock_metric,
        mock_columns,
        mock_subheader,
        mock_markdown,
        mock_title,
    ):
        """Test app initialization and component display"""

        mock_set_page_config.assert_called_once()
        mock_title.assert_called_once_with("CodeGen Hub")
        mock_markdown.assert_called()
        mock_display_sidebar.assert_called_once()

    @patch("streamlit.session_state", new_callable=dict)
    def test_session_state_initialization(self, mock_session_state):
        """Test session state variable initialization"""

        self.assertIn("datasets", mock_session_state)
        self.assertIn("trained_models", mock_session_state)
        self.assertIn("training_logs", mock_session_state)
        self.assertIn("training_progress", mock_session_state)

        self.assertEqual(mock_session_state["datasets"], {})
        self.assertEqual(mock_session_state["trained_models"], {})
        self.assertEqual(mock_session_state["training_logs"], [])
        self.assertEqual(mock_session_state["training_progress"], {})

    @patch("streamlit.metric")
    @patch("streamlit.session_state", new_callable=dict)
    def test_platform_statistics_display(self, mock_session_state, mock_metric):
        """Test platform statistics metrics display"""
        mock_session_state["datasets"] = {"dataset1": "data1", "dataset2": "data2"}
        mock_session_state["trained_models"] = {"model1": "trained_model1"}
        mock_session_state["training_progress"] = {
            "job1": {"status": "running"},
            "job2": {"status": "completed"},
            "job3": {"status": "running"},
        }

        mock_metric.assert_any_call("Datasets Available", 2)
        mock_metric.assert_any_call("Trained Models", 1)
        mock_metric.assert_any_call("Active Training Jobs", 2)

    @patch("streamlit.info")
    def test_getting_started_section_display(self, mock_info):
        """Test 'Getting Started' section instructions display"""

        expected_messages = [
            "1. 📊 Start by uploading or selecting a Python code dataset in the **Dataset Management** section.",
            "2. 🛠️ Configure and train your model in the **Model Training** section.",
            "3. 💡 Generate code predictions using your trained models in the **Code Generation** section.",
            "4. 🔄 Access your models on Hugging Face Hub for broader use.",
        ]

        for msg in expected_messages:
            mock_info.assert_any_call(msg)
        self.assertEqual(mock_info.call_count, 4)


if __name__ == "__main__":
    unittest.main()
