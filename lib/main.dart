import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';

void main() => runApp(const DuckAIApp());

class DuckAIApp extends StatelessWidget {
  const DuckAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.orange, useMaterial3: true),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final String _baseUrl = "http://192.168.2.4:8000";
  final TextEditingController _duckController = TextEditingController(text: "100");
  Map<String, dynamic>? _prediction;
  bool _isLoading = false;

  Future<void> _uploadCSV() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.custom, allowedExtensions: ['csv']);
    if (result != null) {
      setState(() => _isLoading = true);
      var request = http.MultipartRequest('POST', Uri.parse('$_baseUrl/upload-csv'));
      request.files.add(http.MultipartFile.fromBytes('file', result.files.first.bytes!, filename: 'data.csv'));
      await request.send();
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Data Sync Complete!")));
    }
  }

  Future<void> _runAnalysis() async {
    setState(() => _isLoading = true);
    try {
      final predRes = await http.get(Uri.parse('$_baseUrl/predict?duck_count=${_duckController.text}&money_on_hand=0'));
      setState(() => _prediction = jsonDecode(predRes.body));
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Error: Check Server Connection")));
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("🦆 Duck Farming AI: Predictive Machine Learning")),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            _buildInputSection(),
            const SizedBox(height: 24),
            if (_isLoading) const Center(child: CircularProgressIndicator()),
            if (_prediction != null) _buildFullReport(),
          ],
        ),
      ),
    );
  }

  Widget _buildInputSection() {
    return Column(
      children: [
        TextField(
          controller: _duckController,
          decoration: const InputDecoration(labelText: "Enter Duck Count for Analysis", border: OutlineInputBorder()),
          keyboardType: TextInputType.number
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(child: OutlinedButton.icon(onPressed: _uploadCSV, icon: const Icon(Icons.upload_file), label: const Text("Upload CSV"))),
            const SizedBox(width: 12),
            Expanded(child: ElevatedButton.icon(onPressed: _runAnalysis, icon: const Icon(Icons.analytics), label: const Text("Run Prediction"), style: ElevatedButton.styleFrom(backgroundColor: Colors.orange, foregroundColor: Colors.white))),
          ],
        )
      ],
    );
  }

  Widget _buildFullReport() {
    var ops = _prediction!['operational_forecast'];
    var fin = _prediction!['financial_forecast'];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("📊 Operational & Financial Report", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const Divider(),
        Row(
          children: [
            _dataCard("Feeds Consumed", "${ops['commercial_feeds_kg']} KG", "Cost: ₱${fin['expenses']}", Icons.restaurant, Colors.brown),
            _dataCard("Waste Produced", "${ops['waste_pellets_kg']} KG", "${ops['restaurants_needed']} Restaurants", Icons.recycling, Colors.green),
          ],
        ),
        Row(
          children: [
            _dataCard("Total Eggs", "${ops['predicted_eggs']}", "Daily Production", Icons.egg, Colors.amber),
            _dataCard("Egg Price", "₱7.00", "Current Market Rate", Icons.sell, Colors.blue),
          ],
        ),
        const SizedBox(height: 16),
        const Text("💰 Profit & Loss Summary", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        const Divider(),
        _financeRow("Total Income", "₱${fin['income']}", Colors.green),
        _financeRow("Total Expenses", "- ₱${fin['expenses']}", Colors.red),
        const Divider(thickness: 2),
        _financeRow("Net Income", "₱${fin['net_income']}", Colors.blue, isBold: true),
      ],
    );
  }

  Widget _dataCard(String title, String value, String subValue, IconData icon, Color color) {
    return Expanded(
      child: Card(
        elevation: 2,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Icon(icon, color: color, size: 30),
              const SizedBox(height: 8),
              Text(title, style: const TextStyle(fontSize: 14, color: Colors.grey)),
              Text(value, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              Text(subValue, style: const TextStyle(fontSize: 12, color: Colors.blueGrey)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _financeRow(String label, String amount, Color color, {bool isBold = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 16, fontWeight: isBold ? FontWeight.bold : FontWeight.normal)),
          Text(amount, style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color)),
        ],
      ),
    );
  }
}