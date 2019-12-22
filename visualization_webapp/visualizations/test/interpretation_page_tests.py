import unittest

from django.test import RequestFactory
from django.conf import settings

from visualization_webapp.visualizations.views import interpretation_page


class IntepretationPageTests(unittest.TestCase):
    def setUp(self):
        # Every test needs access to the request factory.

        settings.configure()
        self.factory = RequestFactory()

    def test_should_redirect_to_page(self):
        request = self.factory.get('/interpretation')
        response = interpretation_page(request)
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()